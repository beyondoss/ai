//! Sandbox-owned jobs: a command that lives in the sandbox, not in the connection that started it.
//!
//! A v1.2 stream makes the connection a command's lifetime, which is right for a command someone is
//! waiting on and wrong for a job. A replica that is reaped, deployed or fails over closes every
//! stream it holds, so a dev server started in the background died with whichever replica happened
//! to be serving the session. A job here is owned by the **sandbox**: it runs until it exits, is
//! killed, or its session is ended, and any replica serving the session can list it, read its output
//! from an offset, and kill it.
//!
//! [`SandboxJobs`] is that contract. [`ShellJobs`] implements it over plain exec commands — `sh`,
//! `setsid`, `mkfifo`, `tee`, `dd`, `wc`, `base64`, `date` — so it works over every exec protocol
//! version and every vendor with no change on their side, busybox included. (An endpoint can offer
//! the same operations natively; the baseline is what every endpoint gets.)
//!
//! ## Layout
//!
//! ```text
//! /tmp/beyond-jobs/               1777, like /tmp: any uid in the sandbox can create its scope
//!   <scope>/                      0700: one session's jobs (sha256 of tenant + session id)
//!     <id>/                       0700: one job (16 random hex digits, chosen by the agent)
//!       cmd                       the command as notices quote it (one line)
//!       started                   epoch seconds
//!       sup                       the supervisor's pid
//!       pgid                      the job's process group (it leads its own session)
//!       out.<offset>              its combined output, in chunks named by their first byte's offset
//!       exit                      the shell's exit status (128 + N for signal N), once it exits
//!       end                       the same status, once the group is empty: the job is over
//!       keep                      "<job number> <kind>": the sandbox owns it now
//!       seen                      the offset of the output the model has been shown
//!       o, s                      the FIFOs the output flows through
//!       notifier                  this supervisor can notify (v1.4); `keep` refuses notify without it
//!       notify                    0600: the curl config of the latest keep's notify (url, headers, body)
//!       ngen                      bumped by every keep once one has carried notify
//!       nlock                     the notifier runs (a directory: `mkdir` is the lock)
//!       left                      the shell exited with its group alive (a held stream's `shell_exit`)
//!       n.out                     the status the last callback attempt got
//! ```
//!
//! Every write that a reader could see half-done goes through a temp file and `mv`. The scope is a
//! digest, so neither another session nor another tenant can name it by accident, and its mode
//! keeps other uids out; a session only ever lists its own scope.
//!
//! ## One job, four processes
//!
//! **Start** (`start` in [`JOB_SCRIPT`]) makes the directory and runs [`SUPERVISOR`] under `setsid`.
//! The supervisor starts the command in a session of its own (so its process group is exactly the
//! command's processes), routes its output into the chunk files through a FIFO, waits for it, and
//! writes `exit` and `end`. Two helpers run beside it, in the supervisor's session and never in the
//! job's group: the **relay**, a loop of `dd bs=4096 count=256` that cuts the output into chunks of at
//! most 1 MiB (not `head -c`, which over-reads a pipe on busybox and loses the bytes between chunks),
//! and the **keeper**, which every half second kills the job if its holder is gone and it is not
//! kept, and every four seconds deletes the oldest chunks beyond the output cap (8 MiB by default).
//! Readers compute where any offset lives from the chunk names, so a deleted chunk is a gap of known
//! size, never a misread.
//!
//! **Held and kept.** A job starts **held**: `start` with `live` = 1 waits for the supervisor, and a
//! `tee` between the job and the relay copies every byte to the start command's own stdout — so the
//! stream that started it (v1.2, or a template transport's pipes) carries its output live, and the
//! supervisor's markers (`\x1ejob-start:<pgid>`, `\x1ejob-shell-exit:<status>`, `\x1ejob-end:<status>`)
//! arrive on its stderr. If that stream's process dies — the endpoint kills it when the connection
//! drops — the keeper kills the job: a command someone was waiting on dies with its caller, as over
//! v1.2. **Keeping** it (`keep`, which records the job number) hands it to the sandbox: from then on
//! a dropped stream costs only the live copy (`tee` ignores `SIGPIPE` and goes on feeding the relay),
//! and the job runs until it exits or is killed. A job can also start kept (`live` = 0), for an
//! endpoint that cannot stream (v1.1); its output is then read by polling ([`poll_follow`]).
//!
//! **Kill** sends `SIGTERM` to every group named, waits up to the grace for them to go, sends
//! `SIGKILL`, and reports any process still alive after a second more (one the sandbox user may not
//! signal). The supervisor notices the empty group within half a second and records `end`, after
//! which nothing ever signals that group number again: `kill` refuses a job whose `end` exists.
//!
//! ## Job notify (v1.4)
//!
//! Offered only when the endpoint advertises it for the baseline (its deployment's setting that
//! kept jobs survive idling), the probe finds `curl`, and standard input reaches the sandbox.
//! `keep` with notify takes the URL and token on **standard input**, never in an argument, and
//! writes them (escaped) into the 0600 `notify` curl config, then bumps `ngen`. It never sends
//! anything itself: the supervisor's **notifier** does, a loop the keeper starts (in the
//! supervisor's session, so outside the group v1.1 kills once `keep` has answered) when `ngen`
//! first appears, or the supervisor starts just before it records the end. The notifier feeds
//! what it sees (`ngen` changing, `left`, `end`, the directory's removal, an attempt's status) to
//! the v1.4 kernel's shell twin ([`exec_notify::SH`]) and does what it says. Each attempt is
//! exactly `curl -q -sS -K notify -o /dev/null -w '%{http_code}' --max-time 30 --proto
//! =http,https`, run in the background so a new keep can abandon it, its status read from `n.out`
//! (`000` is a connection error). The end's status is on disk before the notifier can report it.
//! The notifier exits once the job has ended and nothing is owed, in flight or waiting, or once the
//! job is removed.
//!
//! ## What is not covered
//!
//! A process that leaves the job's session (`setsid`, a daemon that double-forks) is not the job's,
//! exactly as with a v1.2 endpoint. Its output, if it kept the job's stdout, is cut off two seconds
//! after the job's group empties. A template transport whose far side does not die with its client
//! (`docker exec`, measured — see `far_kill`) never loses its holder when the agent dies abruptly, so
//! a held job there runs until the next replica reattaches the session and kills it as stale.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use sha2::Digest as _;

use crate::tools::exec::CommandRunner;

/// Where every scope lives, in the sandbox.
pub const JOBS_ROOT: &str = "/tmp/beyond-jobs";

/// How much of a job's output the sandbox keeps, at least: older chunks are deleted past it. The
/// model is shown a few tens of KB of any output; this is what a replica that reattaches after a
/// long absence can still read back, and what a job printing forever costs the sandbox's `/tmp`.
pub const OUTPUT_CAP: u64 = 8 * 1024 * 1024;

/// The most output one `read` returns. Base64'd it must fit the smallest answer a runner returns
/// whole: a template transport and this host keep 256 KiB of each stream (`STREAM_HEAD` +
/// `STREAM_TAIL`), a v1.1 endpoint 16 MiB.
pub const READ_MAX: usize = 128 * 1024;

/// How long an operation other than a held start may take: one round trip and a few forks.
const OP_TIMEOUT: Duration = Duration::from_secs(30);

/// The `SIGTERM` grace a kill gives a job before `SIGKILL` — the local runner's
/// (`bash_proc::TERM_GRACE`), so a job is stopped the same way wherever it runs.
const KILL_GRACE_MS: u64 = 300;

/// A polling follower's first and longest wait between reads ([`poll_follow`]).
const POLL_MIN: Duration = Duration::from_millis(250);
const POLL_MAX: Duration = Duration::from_secs(2);

/// Consecutive failed reads after which a polling follower gives up.
const POLL_FAILURES: u32 = 5;

/// How long a held job's shell-exit marker waits for output still in flight on the other pipe — the
/// local runner's 100 ms (`bash_proc::SETTLE`), so what the shell printed lands before the leftovers
/// are reported.
const SHELL_EXIT_SETTLE: Duration = Duration::from_millis(100);

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
    fn parse(state: &str, code: &str) -> Option<Self> {
        let code = || code.trim().parse::<i32>().ok();
        Some(match state {
            "running" => Self::Running,
            "exited" => Self::ShellExited(code()?),
            "ended" => Self::Ended(code()?),
            "lost" => Self::Lost,
            "gone" => Self::Gone,
            _ => return None,
        })
    }

    pub fn ended(self) -> bool {
        matches!(self, Self::Ended(_) | Self::Lost | Self::Gone)
    }
}

/// One job, as `list` reports it.
#[derive(Clone, Debug)]
pub struct JobInfo {
    pub id: JobId,
    pub status: JobStatus,
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
    pub offset: u64,
    pub end: u64,
    pub data: Vec<u8>,
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

/// `keep`'s `notify` (v1.4): where the sandbox reports a kept job's state changes, and the token it
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
    /// Whether this sandbox can run jobs at all. Probed once; a probe that could not reach the
    /// sandbox is retried next time.
    async fn available(&self) -> bool;

    /// Whether a job can start **held** ([`Self::run_held`]): its start streams.
    async fn holds(&self) -> bool;

    /// Start a held job and stream it until it ends, returning its status. `events` hears
    /// [`JobEvent::Started`] once it runs. Dropping the future drops the hold: a job not kept by
    /// then is killed by the sandbox; a kept one runs on. An error after `Started` means the stream
    /// broke, not that the job ended — a kept job is followed with [`Self::follow`] from there.
    async fn run_held(
        &self,
        scope: &JobScope,
        id: &JobId,
        spec: &JobSpec,
        events: EventSink<'_>,
    ) -> std::io::Result<i32>;

    /// Start a job already kept (no hold), returning its process group once it runs.
    async fn start_kept(
        &self,
        scope: &JobScope,
        id: &JobId,
        spec: &JobSpec,
    ) -> std::io::Result<u32>;

    /// Follow a job's output from `from` until it ends, returning its status.
    async fn follow(
        &self,
        scope: &JobScope,
        id: &JobId,
        from: u64,
        events: EventSink<'_>,
    ) -> std::io::Result<i32> {
        poll_follow(self, scope, id, from, events).await
    }

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

    /// Whether `keep` can carry `notify` here (v1.4): the endpoint offers it natively, or for the
    /// baseline. Probed with the rest of the job capability.
    async fn offers_notify(&self) -> bool {
        false
    }

    /// [`Self::keep`] carrying `notify` (v1.4). A sandbox that refuses the `notify` (`400`: it fails
    /// the spec's grammar, or this job cannot be notified about) records nothing, and the error's
    /// kind is [`std::io::ErrorKind::InvalidInput`]; the job is then kept with a plain `keep`, or not
    /// at all. A plain `keep` afterwards removes the job's `notify`.
    async fn keep_notify(
        &self,
        _scope: &JobScope,
        _id: &JobId,
        _num: u64,
        _kind: &str,
        _seen: u64,
        _notify: &JobNotify,
    ) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "this sandbox does not offer job notify (v1.4)",
        ))
    }

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

/// [`SandboxJobs::follow`] by polling [`SandboxJobs::read`]: every 250 ms while output arrives,
/// backing off to every 2 s while it does not. For a sandbox whose jobs cannot be streamed after
/// their start — every baseline sandbox once the replica that started a job has gone.
pub async fn poll_follow<J: SandboxJobs + ?Sized>(
    jobs: &J,
    scope: &JobScope,
    id: &JobId,
    from: u64,
    events: EventSink<'_>,
) -> std::io::Result<i32> {
    let mut offset = from;
    let mut wait = POLL_MIN;
    let mut failures = 0;
    let mut shell_told = false;
    loop {
        let chunk = match jobs.read(scope, id, offset, READ_MAX).await {
            Ok(c) => {
                failures = 0;
                c
            }
            Err(e) => {
                failures += 1;
                if failures >= POLL_FAILURES {
                    return Err(e);
                }
                tokio::time::sleep(wait).await;
                wait = (wait * 2).min(POLL_MAX);
                continue;
            }
        };
        if chunk.status == JobStatus::Gone {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "the job is no longer in the sandbox",
            ));
        }
        if chunk.offset > offset {
            events(JobEvent::Gap(chunk.offset - offset));
            offset = chunk.offset;
        }
        if !chunk.data.is_empty() {
            events(JobEvent::Output(&chunk.data));
            offset += chunk.data.len() as u64;
            wait = POLL_MIN;
            if offset < chunk.end {
                continue;
            }
        }
        match chunk.status {
            JobStatus::ShellExited(code) if !shell_told => {
                shell_told = true;
                events(JobEvent::ShellExit(code));
            }
            // Everything it printed has been read (or there is nothing left to read).
            JobStatus::Ended(code) if offset >= chunk.end || chunk.data.is_empty() => {
                return Ok(code);
            }
            JobStatus::Ended(_) => continue,
            JobStatus::Lost => {
                return Err(std::io::Error::other(
                    "the job's supervisor in the sandbox is gone, and how it ended was not recorded",
                ));
            }
            _ => {}
        }
        tokio::time::sleep(wait).await;
        wait = (wait * 2).min(POLL_MAX);
    }
}

/// The baseline: every operation is one plain exec command running [`JOB_SCRIPT`].
pub struct ShellJobs {
    runner: Arc<dyn CommandRunner>,
    /// What the sandbox has — shared by every `ShellJobs` over one runner.
    probed: Arc<tokio::sync::OnceCell<Baseline>>,
    cap: u64,
    /// Whether the endpoint offers notify for the baseline (v1.4): its deployment's setting that
    /// kept jobs survive idling, as its probe answer says. Nothing else enables notify here.
    advertised: Advertised,
}

/// What the baseline's probe found in the sandbox.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Baseline {
    /// Every tool the job scripts need is there.
    pub jobs: bool,
    /// `curl` is there and the runner delivers standard input: the supervisor can notify (v1.4).
    pub notifier: bool,
}

/// Where the baseline learns whether the endpoint offers it notify.
#[derive(Clone)]
enum Advertised {
    /// No endpoint says so: a template transport, this host.
    Never,
    /// Told outright. For tests.
    Fixed(bool),
    /// The exec endpoint's answer to the job probe (`{"notify": "v1.4"}` and no `jobs`).
    Endpoint(Arc<super::HttpExecRunner>),
}

impl ShellJobs {
    pub fn new(runner: Arc<dyn CommandRunner>) -> Self {
        Self::with_probe(runner, Arc::default())
    }

    /// Over `runner`, sharing its probe result through `probed`.
    pub fn with_probe(
        runner: Arc<dyn CommandRunner>,
        probed: Arc<tokio::sync::OnceCell<Baseline>>,
    ) -> Self {
        Self {
            runner,
            probed,
            cap: OUTPUT_CAP,
            advertised: Advertised::Never,
        }
    }

    /// Over an exec endpoint, which says (in its job probe's answer) whether it offers notify for
    /// the baseline.
    pub(crate) fn over_endpoint(
        runner: Arc<super::HttpExecRunner>,
        probed: Arc<tokio::sync::OnceCell<Baseline>>,
    ) -> Self {
        let mut me = Self::with_probe(runner.clone(), probed);
        me.advertised = Advertised::Endpoint(runner);
        me
    }

    /// As if the endpoint's deployment said (or denied) that kept jobs survive idling, which is
    /// what offers notify to the baseline. For tests, and for a transport with no endpoint to say.
    pub fn with_notify_advertised(mut self, advertised: bool) -> Self {
        self.advertised = Advertised::Fixed(advertised);
        self
    }

    /// What the sandbox has, probing it the first time. A probe that could not reach it finds
    /// nothing and is retried next time.
    pub async fn baseline(&self) -> Baseline {
        self.probed
            .get_or_try_init(|| async {
                let argv = Self::argv(&JobScope::under(JOBS_ROOT, "probe"), "probe", Vec::new());
                let r = self
                    .runner
                    .run_with_stdin("sh", &argv, None, OP_TIMEOUT, b"stdin-probe\n")
                    .await?;
                let mut lines = r.stdout.lines().map(str::trim);
                let jobs = r.code == Some(0) && lines.next() == Some("ok");
                let rest: Vec<&str> = lines.collect();
                if !jobs {
                    tracing::info!(
                        answer = %r.stdout.trim(),
                        "the sandbox cannot run jobs of its own (a tool is missing); `bash` jobs \
                         live in their streams there"
                    );
                }
                Ok::<_, std::io::Error>(Baseline {
                    jobs,
                    notifier: jobs && rest.contains(&"curl") && rest.contains(&"stdin"),
                })
            })
            .await
            .copied()
            .unwrap_or_default()
    }

    /// A different output cap than [`OUTPUT_CAP`]. For tests.
    pub fn with_output_cap(mut self, cap: u64) -> Self {
        self.cap = cap;
        self
    }

    fn argv(scope: &JobScope, op: &str, args: Vec<String>) -> Vec<String> {
        let mut argv = Vec::with_capacity(6 + args.len());
        argv.extend([
            "-c".to_string(),
            JOB_SCRIPT.to_string(),
            "sh".to_string(),
            op.to_string(),
            scope.root.clone(),
            scope.name.clone(),
        ]);
        argv.extend(args);
        argv
    }

    /// Run one operation; its stdout when it succeeds.
    async fn op(&self, scope: &JobScope, op: &str, args: Vec<String>) -> std::io::Result<String> {
        self.op_fed(scope, op, args, None).await
    }

    /// [`Self::op`], with `stdin` fed to the script. Exit 4 is a refusal of the request as written
    /// ([`std::io::ErrorKind::InvalidInput`]).
    async fn op_fed(
        &self,
        scope: &JobScope,
        op: &str,
        args: Vec<String>,
        stdin: Option<&[u8]>,
    ) -> std::io::Result<String> {
        let argv = Self::argv(scope, op, args);
        let r = match stdin {
            Some(bytes) => {
                self.runner
                    .run_with_stdin("sh", &argv, None, OP_TIMEOUT, bytes)
                    .await?
            }
            None => self.runner.run("sh", &argv, None, OP_TIMEOUT).await?,
        };
        if r.code == Some(4) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("sandbox job `{op}` refused: this job's supervisor cannot notify"),
            ));
        }
        if r.truncated {
            return Err(std::io::Error::other(format!(
                "sandbox job `{op}`: the answer was cut short by the runner's capture window"
            )));
        }
        if r.code != Some(0) {
            return Err(std::io::Error::other(format!(
                "sandbox job `{op}` failed ({}): {}",
                r.code
                    .map_or_else(|| "no exit code".to_string(), |c| format!("exit {c}")),
                r.stderr.trim()
            )));
        }
        Ok(r.stdout)
    }

    fn start_args(&self, id: &JobId, live: bool, spec: &JobSpec) -> Vec<String> {
        let mut args = vec![
            id.0.clone(),
            if live { "1" } else { "0" }.to_string(),
            self.cap.to_string(),
            spec.cwd.clone().unwrap_or_default(),
            one_line(&spec.label),
            SUPERVISOR.clone(),
            spec.program.clone(),
        ];
        args.extend(spec.args.iter().cloned());
        args
    }
}

/// `label` on one line, as the `cmd` file holds it.
fn one_line(label: &str) -> String {
    label
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}

#[async_trait]
impl SandboxJobs for ShellJobs {
    async fn available(&self) -> bool {
        self.baseline().await.jobs
    }

    async fn holds(&self) -> bool {
        self.runner.attaches().await
    }

    async fn run_held(
        &self,
        scope: &JobScope,
        id: &JobId,
        spec: &JobSpec,
        events: EventSink<'_>,
    ) -> std::io::Result<i32> {
        let mut argv = Self::argv(scope, "start", Vec::new());
        argv.extend(self.start_args(id, true, spec));
        let markers = std::sync::Mutex::new(Markers::default());
        let (shell_tx, mut shell_rx) = tokio::sync::watch::channel(None::<i32>);
        let on_stdout = |bytes: &[u8]| events(JobEvent::Output(bytes));
        let on_stderr = |bytes: &[u8]| {
            let mut m = markers.lock().unwrap_or_else(|e| e.into_inner());
            for marker in m.feed(bytes) {
                match marker {
                    Marker::Start(pgid) => events(JobEvent::Started(pgid)),
                    Marker::ShellExit(code) => {
                        let _ = shell_tx.send(Some(code));
                    }
                    Marker::End(_) => {}
                }
            }
        };
        let (done_tx, done_rx) = tokio::sync::watch::channel(false);
        let run = async {
            let r = self
                .runner
                .run_attached("sh", &argv, None, &on_stdout, &on_stderr)
                .await;
            let _ = done_tx.send(true);
            r
        };
        // The shell-exit marker is reported after the settle, so output in flight on stdout lands
        // first — unless the stream ends sooner.
        let shell = async {
            let mut done = done_rx.clone();
            tokio::select! {
                code = async { shell_rx.wait_for(Option::is_some).await.ok().and_then(|c| *c) } => {
                    tokio::time::sleep(SHELL_EXIT_SETTLE).await;
                    if let Some(code) = code {
                        events(JobEvent::ShellExit(code));
                    }
                }
                () = async { let _ = done.wait_for(|d| *d).await; } => {}
            }
        };
        let (result, ()) = tokio::join!(run, shell);
        let result = result?;
        let m = markers.into_inner().unwrap_or_else(|e| e.into_inner());
        match (m.end, m.started) {
            (Some(code), _) => Ok(code),
            (None, false) => Err(std::io::Error::other(format!(
                "the sandbox did not start the job ({}): {}",
                exit_text(&result),
                m.diagnostics().trim()
            ))),
            (None, true) => Err(std::io::Error::other(format!(
                "the job's stream ended before its end was reported ({}){}",
                exit_text(&result),
                match m.diagnostics().trim() {
                    "" => String::new(),
                    d => format!(": {d}"),
                }
            ))),
        }
    }

    async fn start_kept(
        &self,
        scope: &JobScope,
        id: &JobId,
        spec: &JobSpec,
    ) -> std::io::Result<u32> {
        let out = self
            .op(scope, "start", self.start_args(id, false, spec))
            .await?;
        out.trim().parse().map_err(|_| {
            std::io::Error::other(format!(
                "the sandbox did not report the job's group: {out:?}"
            ))
        })
    }

    async fn read(
        &self,
        scope: &JobScope,
        id: &JobId,
        offset: u64,
        max: usize,
    ) -> std::io::Result<ReadChunk> {
        let out = self
            .op(
                scope,
                "read",
                vec![id.0.clone(), offset.to_string(), max.to_string()],
            )
            .await?;
        parse_read(&out)
    }

    async fn keep(
        &self,
        scope: &JobScope,
        id: &JobId,
        num: u64,
        kind: &str,
        seen: u64,
    ) -> std::io::Result<()> {
        self.op(
            scope,
            "keep",
            vec![
                id.0.clone(),
                num.to_string(),
                one_word(kind),
                seen.to_string(),
            ],
        )
        .await
        .map(drop)
    }

    async fn offers_notify(&self) -> bool {
        let advertised = match &self.advertised {
            Advertised::Never => false,
            Advertised::Fixed(a) => *a,
            Advertised::Endpoint(r) => r.job_offer().await.baseline_notify,
        };
        advertised && self.baseline().await.notifier
    }

    /// The `notify` reaches the sandbox on standard input, never in an argument, and is kept in a
    /// 0600 curl config the job's supervisor reads for each attempt.
    async fn keep_notify(
        &self,
        scope: &JobScope,
        id: &JobId,
        num: u64,
        kind: &str,
        seen: u64,
        notify: &JobNotify,
    ) -> std::io::Result<()> {
        if let Err(why) = exec_notify::check_notify(&notify.url, &notify.token) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("refused notify: {why}"),
            ));
        }
        if !self.offers_notify().await {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "this sandbox's jobs cannot notify",
            ));
        }
        let stdin = format!("{}\n{}\n", notify.url, notify.token);
        self.op_fed(
            scope,
            "keep",
            vec![
                id.0.clone(),
                num.to_string(),
                one_word(kind),
                seen.to_string(),
                "n".to_string(),
            ],
            Some(stdin.as_bytes()),
        )
        .await
        .map(drop)
    }

    async fn seen(&self, scope: &JobScope, id: &JobId, offset: u64) -> std::io::Result<()> {
        self.op(scope, "seen", vec![id.0.clone(), offset.to_string()])
            .await
            .map(drop)
    }

    async fn kill(
        &self,
        scope: &JobScope,
        ids: &[JobId],
        remove: bool,
    ) -> std::io::Result<Vec<(JobId, String)>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut args = vec![
            KILL_GRACE_MS.to_string(),
            if remove { "1" } else { "0" }.to_string(),
        ];
        args.extend(ids.iter().map(|id| id.0.clone()));
        let out = self.op(scope, "kill", args).await?;
        Ok(out
            .lines()
            .filter_map(|l| {
                let mut f = l.split(' ');
                (f.next() == Some("survived"))
                    .then(|| Some((JobId::parse(f.next()?)?, describe_survivors(f.next()?))))?
            })
            .collect())
    }

    async fn list(&self, scope: &JobScope) -> std::io::Result<Vec<JobInfo>> {
        let out = self.op(scope, "list", Vec::new()).await?;
        Ok(out.lines().filter_map(parse_list_line).collect())
    }

    async fn remove(&self, scope: &JobScope, ids: &[JobId]) -> std::io::Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        self.op(scope, "rm", ids.iter().map(|id| id.0.clone()).collect())
            .await
            .map(drop)
    }

    async fn wipe(&self, scope: &JobScope) -> std::io::Result<Vec<String>> {
        let out = self
            .op(scope, "wipe", vec![KILL_GRACE_MS.to_string()])
            .await?;
        Ok(out
            .lines()
            .filter_map(|l| l.strip_prefix("survived - ").map(describe_survivors))
            .collect())
    }
}

/// A kind as `keep` stores it: one word.
fn one_word(kind: &str) -> String {
    let w: String = kind
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if w.is_empty() { "-".to_string() } else { w }
}

/// `"12,34"` (the pids `kill` could not stop) as the model is told.
pub(crate) fn describe_survivors(pids: &str) -> String {
    let pids: Vec<&str> = pids.split(',').filter(|p| !p.is_empty()).collect();
    match pids.as_slice() {
        [] => "a process in the sandbox".to_string(),
        [one] => format!("process {one} in the sandbox"),
        many => format!("processes {} in the sandbox", many.join(", ")),
    }
}

fn exit_text(r: &crate::tools::exec::ExecResult) -> String {
    match (r.code, r.signal, r.timed_out) {
        (_, _, true) => "the exec endpoint timed it out".to_string(),
        (Some(c), _, _) => format!("exit {c}"),
        (None, Some(s), _) => format!("signal {s}"),
        (None, None, _) => "no exit status".to_string(),
    }
}

/// `read`'s answer: `<state> <code> <offset> <end>`, then the base64 of the bytes.
fn parse_read(out: &str) -> std::io::Result<ReadChunk> {
    let bad = || std::io::Error::other(format!("unreadable job read: {:?}", excerpt(out)));
    let mut lines = out.splitn(2, '\n');
    let head = lines.next().unwrap_or("").trim();
    if head == "gone" {
        return Ok(ReadChunk {
            status: JobStatus::Gone,
            offset: 0,
            end: 0,
            data: Vec::new(),
        });
    }
    let f: Vec<&str> = head.split(' ').collect();
    let [state, code, offset, end] = f.as_slice() else {
        return Err(bad());
    };
    let status = JobStatus::parse(state, code).ok_or_else(bad)?;
    let data = lines.next().unwrap_or("").trim();
    Ok(ReadChunk {
        status,
        offset: offset.parse().map_err(|_| bad())?,
        end: end.parse().map_err(|_| bad())?,
        data: base64::engine::general_purpose::STANDARD
            .decode(data)
            .map_err(|_| bad())?,
    })
}

/// One `list` line: `<id> <state> <code> <num|-> <kind|-> <seen> <started> <ended> <label…>`.
fn parse_list_line(line: &str) -> Option<JobInfo> {
    let mut f = line.splitn(9, ' ');
    let id = JobId::parse(f.next()?)?;
    let status = JobStatus::parse(f.next()?, f.next()?)?;
    let num = f.next()?;
    let kind = f.next()?;
    let seen = f.next()?.parse().unwrap_or(0);
    let started = f.next()?.parse().unwrap_or(0);
    let ended = f.next()?.parse().unwrap_or(0);
    let label = f.next().unwrap_or("").to_string();
    Some(JobInfo {
        id,
        status,
        // `keep` writes the file whole, and a job started kept writes `- background`.
        kept: kind != "-" || num != "-",
        num: num.parse().ok(),
        kind: (kind != "-").then(|| kind.to_string()),
        seen,
        started,
        ended,
        label,
    })
}

fn excerpt(s: &str) -> String {
    s.chars().take(200).collect()
}

/// A supervisor marker, heard on a held start's stderr.
#[derive(Debug, PartialEq, Eq)]
enum Marker {
    Start(u32),
    ShellExit(i32),
    End(i32),
}

/// Splits a held start's stderr into markers and everything else (kept, bounded, for errors).
#[derive(Default)]
struct Markers {
    line: Vec<u8>,
    other: Vec<u8>,
    started: bool,
    end: Option<i32>,
}

/// How much non-marker stderr is kept for an error message.
const MAX_DIAGNOSTICS: usize = 4096;

impl Markers {
    fn feed(&mut self, bytes: &[u8]) -> Vec<Marker> {
        let mut out = Vec::new();
        for &b in bytes {
            if b != b'\n' {
                if self.line.len() < MAX_DIAGNOSTICS {
                    self.line.push(b);
                }
                continue;
            }
            let line = std::mem::take(&mut self.line);
            match parse_marker(&line) {
                Some(m) => {
                    match m {
                        Marker::Start(_) => self.started = true,
                        Marker::End(code) => self.end = Some(code),
                        Marker::ShellExit(_) => {}
                    }
                    out.push(m);
                }
                None => {
                    if self.other.len() < MAX_DIAGNOSTICS {
                        self.other.extend_from_slice(&line);
                        self.other.push(b'\n');
                    }
                }
            }
        }
        out
    }

    fn diagnostics(&self) -> String {
        let mut d = String::from_utf8_lossy(&self.other).into_owned();
        d.push_str(&String::from_utf8_lossy(&self.line));
        d
    }
}

fn parse_marker(line: &[u8]) -> Option<Marker> {
    let line = std::str::from_utf8(line.strip_prefix(b"\x1ejob-")?).ok()?;
    let (kind, value) = line.split_once(':')?;
    let value = value.trim();
    Some(match kind {
        "start" => Marker::Start(value.parse().ok()?),
        "shell-exit" => Marker::ShellExit(value.parse().ok()?),
        "end" => Marker::End(value.parse().ok()?),
        _ => return None,
    })
}

/// Every operation but the supervisor, as `sh -c JOB_SCRIPT sh <op> <root> <scope> <args…>`.
/// Positional parameters throughout: nothing is ever spliced into script text. POSIX `sh` and
/// busybox alike; the helpers use only builtins, and their variables start with `_` so they cannot
/// clobber a caller's.
///
/// - `probe` — `ok`, or the tools that are missing.
/// - `start <id> <live> <cap> <cwd> <label> <supervisor> <program> <args…>` — see the module doc.
///   Live: waits for the supervisor, whose markers arrive on stderr and whose `tee` writes the output
///   to stdout. Not live: the job starts kept, and its group is printed once it runs.
/// - `read <id> <offset> <max>` — `<state> <code> <offset> <end>`, then the base64 of up to `max`
///   bytes from `offset` (or from the oldest byte kept, when that is later), within one chunk.
/// - `list` — one line per job: `<id> <state> <code> <num> <kind> <seen> <started> <ended> <label>`.
/// - `keep <id> <num> <kind> <seen>`, `seen <id> <offset>`.
/// - `kill <grace-ms> <remove> <id…>` — `survived <id> <pid,pid>` for each job not stopped. A
///   removed job's directory goes once its supervisor has recorded the end (at most 3 s), so it
///   writes nothing into a directory being deleted.
/// - `rm <id…>` — deletes each job whose group is empty.
/// - `wipe <grace-ms>` — kills every job in the scope and deletes it.
pub const JOB_SCRIPT: &str = r#"umask 077
op=$1 R=$2 S=$2/$3
shift 3
live() {
  kill -0 -"$g" 2>/dev/null || return 1
  [ -d /proc/self ] || return 0
  for _f in /proc/[0-9]*/stat; do
    read -r _t 2>/dev/null <"$_f" || continue
    _t=${_t##*) }
    set -- $_t
    [ "$3" = "$g" ] && [ "$1" != Z ] && return 0
  done
  return 1
}
members() {
  M=
  [ -d /proc/self ] || return 0
  for _f in /proc/[0-9]*/stat; do
    read -r _t 2>/dev/null <"$_f" || continue
    _p=${_f#/proc/} _p=${_p%/stat} _t=${_t##*) }
    set -- $_t
    [ "$3" = "$g" ] && [ "$1" != Z ] && M=$M${M:+,}$_p
  done
}
alive() {
  kill -0 "$1" 2>/dev/null || return 1
  [ -r "/proc/$1/stat" ] || return 0
  read -r _t 2>/dev/null <"/proc/$1/stat" || return 1
  _t=${_t##*) }
  [ "${_t%% *}" != Z ]
}
num() { _n=${1##*.}; _n=${_n#"${_n%%[!0]*}"}; N=${_n:-0}; }
state() {
  st=running c=-
  if [ -e "$j/end" ]; then st=ended; read -r c _e <"$j/end"
  elif [ -e "$j/exit" ]; then st=exited; read -r c <"$j/exit"
  elif [ -e "$j/sup" ]; then read -r _s <"$j/sup"; alive "$_s" || st=lost
  else st=lost
  fi
}
gone() {
  _i=0
  while [ ! -e "$j/end" ] && [ -e "$j/sup" ] && [ $_i -lt 60 ]; do
    read -r _s <"$j/sup"; alive "$_s" || break
    sleep 0.05 2>/dev/null || sleep 1; _i=$((_i + 1))
  done
  rm -rf "$j" 2>/dev/null || { sleep 0.1 2>/dev/null; rm -rf "$j" 2>/dev/null; }
  return 0
}
stop() {
  for g in $gs; do kill -TERM -"$g" 2>/dev/null; done
  _i=0
  while [ $_i -lt $(($1 / 50)) ]; do
    _a=; for g in $gs; do live && { _a=1; break; }; done
    [ -n "$_a" ] || return 0
    sleep 0.05 2>/dev/null || sleep 1; _i=$((_i + 1))
  done
  for g in $gs; do kill -KILL -"$g" 2>/dev/null; done
  _i=0
  while [ $_i -lt 20 ]; do
    _a=; for g in $gs; do live && { _a=1; break; }; done
    [ -n "$_a" ] || return 0
    sleep 0.05 2>/dev/null || sleep 1; _i=$((_i + 1))
  done
}
case $op in
probe)
  m=
  for c in setsid mkfifo tee dd wc base64 date; do
    command -v "$c" >/dev/null 2>&1 || m="$m $c"
  done
  if [ -z "$m" ]; then echo ok; else echo "missing$m"; fi
  command -v curl >/dev/null 2>&1 && echo curl
  IFS= read -r _s 2>/dev/null && [ "$_s" = stdin-probe ] && echo stdin
  true
  ;;
start)
  j=$S/$1 lv=$2 cap=$3 d=$4 lb=$5 sup=$6
  shift 6
  if [ ! -d "$R" ]; then mkdir -p "$R" 2>/dev/null && chmod 1777 "$R" 2>/dev/null; fi
  mkdir -p "$S" 2>/dev/null
  mkdir "$j" || exit 125
  printf '%s\n' "$lb" >"$j/cmd"
  date +%s >"$j/started"
  if [ "$lv" = 1 ]; then
    setsid sh -c "$sup" sh "$j" 1 "$cap" "$d" "$@" &
    wait $!
    exit $?
  fi
  printf -- '- background\n' >"$j/keep"
  setsid sh -c "$sup" sh "$j" 0 "$cap" "$d" "$@" </dev/null >/dev/null 2>&1 &
  _i=0
  while [ ! -e "$j/pgid" ] && [ $_i -lt 500 ]; do sleep 0.01 2>/dev/null || sleep 1; _i=$((_i + 1)); done
  [ -e "$j/pgid" ] || exit 125
  cat "$j/pgid"
  ;;
read)
  j=$S/$1 o=$2 m=$3
  [ -d "$j" ] || { echo gone; exit 0; }
  state
  f= e=0 z=
  for x in "$j"/out.*; do
    [ -e "$x" ] || break
    num "$x"
    if [ "$N" -le "$o" ] || [ -z "$f" ]; then f=$x fs=$N; fi
    z=$x zs=$N
  done
  [ -n "$z" ] && e=$((zs + $(wc -c <"$z")))
  if [ -n "$f" ] && [ "$fs" -gt "$o" ]; then o=$fs; fi
  printf '%s %s %s %s\n' "$st" "$c" "$o" "$e"
  if [ -n "$f" ]; then
    dd if="$f" iflag=skip_bytes,count_bytes skip=$((o - fs)) count="$m" status=none | base64 -w0
  fi
  echo
  ;;
list)
  for j in "$S"/*; do
    [ -d "$j" ] || continue
    state
    k=- kd=- sn=0 t=0 ed=0 lb=
    [ -e "$j/end" ] && read -r _c ed <"$j/end"
    [ -e "$j/keep" ] && read -r k kd <"$j/keep"
    [ -e "$j/seen" ] && read -r sn <"$j/seen"
    [ -e "$j/started" ] && read -r t <"$j/started"
    [ -e "$j/cmd" ] && read -r lb <"$j/cmd"
    printf '%s %s %s %s %s %s %s %s %s\n' "${j##*/}" "$st" "$c" "$k" "$kd" "$sn" "$t" "${ed:-0}" "$lb"
  done
  ;;
keep)
  j=$S/$1
  [ -d "$j" ] || exit 3
  if [ "$5" = n ]; then
    [ -e "$j/notifier" ] || exit 4
    [ -e "$j/end" ] && [ ! -d "$j/nlock" ] && exit 4
    IFS= read -r u && IFS= read -r t || exit 4
    q() { printf '%s' "$1" | sed 's/[\\"]/\\&/g'; }
    { printf 'url = "%s"\n' "$(q "$u")"
      printf 'header = "Authorization: Bearer %s"\n' "$(q "$t")"
      printf 'header = "Content-Type: application/json"\n'
      printf 'data-binary = "{\\"type\\":\\"job_ended\\",\\"scope\\":\\"%s\\",\\"id\\":\\"%s\\"}"\n' "${S##*/}" "$1"
    } >"$j/notify.t" || exit 1
    u= t=
  fi
  printf '%s\n' "$4" >"$j/seen.t" && mv -f "$j/seen.t" "$j/seen"
  printf '%s %s\n' "$2" "$3" >"$j/keep.t" && mv -f "$j/keep.t" "$j/keep"
  if [ "$5" = n ]; then mv -f "$j/notify.t" "$j/notify"; elif [ -e "$j/ngen" ]; then rm -f "$j/notify"; fi
  if [ "$5" = n ] || [ -e "$j/ngen" ]; then
    n=0; [ -e "$j/ngen" ] && read -r n <"$j/ngen"
    printf '%s\n' $((n + 1)) >"$j/ngen.t" && mv -f "$j/ngen.t" "$j/ngen"
  fi
  ;;
seen)
  j=$S/$1
  [ -d "$j" ] || exit 3
  printf '%s\n' "$2" >"$j/seen.t" && mv -f "$j/seen.t" "$j/seen"
  ;;
kill)
  gr=$1 rmf=$2
  shift 2
  gs=
  for id in "$@"; do
    j=$S/$id
    [ -e "$j/pgid" ] && [ ! -e "$j/end" ] || continue
    read -r g <"$j/pgid"
    gs="$gs $g"
  done
  stop "$gr"
  for id in "$@"; do
    j=$S/$id
    [ -d "$j" ] || continue
    g=
    [ -e "$j/pgid" ] && [ ! -e "$j/end" ] && read -r g <"$j/pgid"
    if [ -n "$g" ] && live; then members; echo "survived $id $M"
    elif [ "$rmf" = 1 ]; then gone
    fi
  done
  ;;
rm)
  for id in "$@"; do
    j=$S/$id
    g=
    [ -e "$j/pgid" ] && [ ! -e "$j/end" ] && read -r g <"$j/pgid"
    if [ -z "$g" ] || ! live; then rm -rf "$j"; fi
  done
  ;;
wipe)
  gs=
  for j in "$S"/*; do
    [ -e "$j/pgid" ] && [ ! -e "$j/end" ] || continue
    read -r g <"$j/pgid"
    gs="$gs $g"
  done
  stop "$1"
  for g in $gs; do live && { members; echo "survived - $M"; }; done
  for j in "$S"/*; do [ -d "$j" ] && gone; done
  rm -rf "$S" 2>/dev/null
  true
  ;;
*)
  echo "unknown op $op" >&2
  exit 2
  ;;
esac"#;

/// The supervisor, as `setsid sh -c SUPERVISOR sh <dir> <live> <cap> <cwd> <program> <args…>` — see
/// the module doc. `$PPID` at its start is the holder: the held start that is waiting for it. The
/// v1.4 kernel's shell twin ([`exec_notify::SH`]) is spliced in for its notifier.
pub static SUPERVISOR: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SUPERVISOR_TEMPLATE.replacen("#@kernel\n", exec_notify::SH, 1));

const SUPERVISOR_TEMPLATE: &str = r#"j=$1 lv=$2 cap=$3 d=$4
shift 4
w=$PPID
echo $$ >"$j/sup"
printf '1\n' >"$j/notifier"
exec 3>&1 4>&2 >/dev/null 2>/dev/null
live() {
  kill -0 -"$g" 2>/dev/null || return 1
  [ -d /proc/self ] || return 0
  for _f in /proc/[0-9]*/stat; do
    read -r _t 2>/dev/null <"$_f" || continue
    _t=${_t##*) }
    set -- $_t
    [ "$3" = "$g" ] && [ "$1" != Z ] && return 0
  done
  return 1
}
alive() {
  kill -0 "$1" 2>/dev/null || return 1
  [ -r "/proc/$1/stat" ] || return 0
  read -r _t 2>/dev/null <"/proc/$1/stat" || return 1
  _t=${_t##*) }
  [ "${_t%% *}" != Z ]
}
num() { _n=${1##*.}; _n=${_n#"${_n%%[!0]*}"}; N=${_n:-0}; }
say() { ( trap '' PIPE; printf '\036job-%s\n' "$1" >&4 ); }
#@kernel
kdo() {
  for _x in $KX; do
    case $_x in
      a) if [ -n "$cp" ]; then kill "$cp" 2>/dev/null; wait "$cp" 2>/dev/null; fi; cp= ;;
      b*) rm -f "$j/n.out"
        curl -q -sS -K "$j/notify" -o /dev/null -w '%{http_code}' --max-time 30 --proto =http,https >"$j/n.out" 2>/dev/null &
        cp=$! ;;
    esac
  done
}
notifier() {
  mkdir "$j/nlock" 2>/dev/null || return 0
  exec </dev/null >/dev/null 2>&1 3>&- 4>&-
  KN=0 KR=0 KO=0 KS=0 KI= KA=0 KW= KX= ng= cp= now=0 sx= se=
  while :; do
    if [ ! -d "$j" ]; then kstep r "$now"; kdo; return 0; fi
    _g=; [ -e "$j/ngen" ] && read -r _g <"$j/ngen"
    if [ -n "$_g" ] && [ "$_g" != "$ng" ]; then
      ng=$_g
      if [ -e "$j/notify" ]; then kstep k1 "$now"; else kstep k0 "$now"; fi
      kdo
    fi
    if [ -z "$sx" ] && [ -e "$j/left" ]; then sx=1; kstep c "$now"; kdo; fi
    if [ -z "$se" ] && [ -e "$j/end" ]; then se=1; kstep c "$now"; kdo; fi
    if [ -n "$cp" ] && ! kill -0 "$cp" 2>/dev/null; then
      wait "$cp" 2>/dev/null; cp= _st=
      read -r _st <"$j/n.out" 2>/dev/null
      case $_st in [1-5][0-9][0-9]) ;; *) _st=f ;; esac
      _d=$(od -An -N4 -tu4 /dev/urandom 2>/dev/null)
      _d=${_d##* }
      case $_d in ''|*[!0-9]*) _d=0 ;; esac
      kstep e "$now" "$_st" "$_d"; kdo
    fi
    if [ -n "$KW" ] && [ "$KW" -le "$now" ]; then kstep w "$now"; kdo; fi
    [ -n "$se" ] && [ -z "$KI$KW" ] && return 0
    if [ -n "$KI$KW" ]; then _p=0.25 _s=250; else _p=0.5 _s=500; fi
    if sleep "$_p" 2>/dev/null; then now=$((now + _s)); else sleep 1; now=$((now + 1000)); fi
  done
}
mkfifo "$j/o" || exit 125
src=$j/o tp=
if [ "$lv" = 1 ]; then
  mkfifo "$j/s" || exit 125
  src=$j/s
  ( trap '' PIPE; exec tee "$j/s" ) <"$j/o" >&3 4>&- &
  tp=$!
fi
( exec <"$src" 3>&- 4>&-
  cd "$j" || exit 1
  o=0 nl='
'
  while :; do
    p=0000000000000000$o
    p=out.${p#"${p%????????????????}"}
    r=$(dd bs=4096 count=256 2>&1 >"$p")
    n=$(wc -c <"$p")
    [ "$n" -gt 0 ] || { rm -f "$p"; exit 0; }
    o=$((o + n))
    r=${r%%" records in"*}
    r=${r##*"$nl"}
    [ $((${r%%+*} + ${r#*+})) -lt 256 ] && exit 0
  done ) &
rp=$!
setsid sh -c 'if [ -n "$1" ]; then cd "$1" || exit 1; fi; shift; exec "$@"' sh "$d" "$@" </dev/null >"$j/o" 2>&1 3>&- 4>&- &
g=$!
printf '%s\n' "$g" >"$j/pgid.t" && mv -f "$j/pgid.t" "$j/pgid"
say "start:$g"
( exec </dev/null 3>&- 4>&-
  t=0
  while [ -d "$j" ] && [ ! -e "$j/end" ]; do
    if [ ! -e "$j/keep" ] && ! alive "$w"; then kill -KILL -"$g"; fi
    if [ -e "$j/ngen" ] && [ ! -d "$j/nlock" ]; then ( notifier ) & fi
    t=$((t + 1))
    if [ $t -ge 8 ]; then
      t=0 z=
      for x in "$j"/out.*; do z=$x; done
      if [ -e "$z" ]; then
        num "$z"; last=$N
        set -- "$j"/out.*
        while [ $# -gt 1 ]; do
          num "$2"
          [ $((last - N)) -ge "$cap" ] || break
          rm -f "$1"; shift
        done
      fi
    fi
    sleep 0.5 2>/dev/null || sleep 1
  done ) &
wait "$g"
s=$?
printf '%s\n' "$s" >"$j/exit.t" && mv -f "$j/exit.t" "$j/exit"
if live; then
  : >"$j/left"
  say "shell-exit:$s"
  while live; do sleep 0.5 2>/dev/null || sleep 1; done
fi
( sleep 2; kill $rp $tp ) 3>&- 4>&- &
k=$!
wait $rp $tp
kill $k 2>/dev/null
if [ -e "$j/ngen" ] && [ ! -d "$j/nlock" ]; then ( notifier ) 3>&- 4>&- & fi
printf '%s %s\n' "$s" "$(date +%s)" >"$j/end.t" && mv -f "$j/end.t" "$j/end"
say "end:$s"
exit "$s""#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scope_is_a_digest_of_tenant_and_session() {
        let a = JobScope::for_session(Some("t1"), "s1");
        assert_eq!(a.root(), JOBS_ROOT);
        assert_eq!(a.name().len(), 32);
        assert!(a.name().bytes().all(|b| b.is_ascii_hexdigit()));
        assert_eq!(a, JobScope::for_session(Some("t1"), "s1"));
        assert_ne!(a, JobScope::for_session(Some("t2"), "s1"));
        assert_ne!(a, JobScope::for_session(Some("t1"), "s2"));
        // The separator is in the digest: ("t", "1s") and ("t1", "s") are different sessions.
        assert_ne!(
            JobScope::for_session(Some("t"), "1s"),
            JobScope::for_session(Some("t1"), "s")
        );
    }

    #[test]
    fn a_job_id_read_back_is_never_a_path() {
        assert!(JobId::parse("0123456789abcdef").is_some());
        for bad in ["", "..", "a/b", "../x", "A", "a b", &"a".repeat(33)] {
            assert!(JobId::parse(bad).is_none(), "{bad:?}");
        }
        let id = JobId::fresh();
        assert_eq!(id.as_str().len(), 16);
        assert_eq!(JobId::parse(id.as_str()), Some(id));
    }

    #[test]
    fn markers_are_split_from_other_stderr_across_chunks() {
        let mut m = Markers::default();
        assert!(m.feed(b"\x1ejob-sta").is_empty());
        assert_eq!(
            m.feed(b"rt:42\nnoise\n\x1ejob-shell-"),
            vec![Marker::Start(42)]
        );
        assert_eq!(
            m.feed(b"exit:3\n\x1ejob-end:3\n"),
            vec![Marker::ShellExit(3), Marker::End(3)]
        );
        assert!(m.started);
        assert_eq!(m.end, Some(3));
        assert_eq!(m.diagnostics(), "noise\n");
        // A marker-shaped line with an unknown kind or a bad value is not a marker.
        assert!(m.feed(b"\x1ejob-what:1\n\x1ejob-end:x\n").is_empty());
    }

    #[test]
    fn a_read_answer_parses() {
        let r = parse_read("ended 0 7 42\naGVsbG8=\n").unwrap();
        assert_eq!(r.status, JobStatus::Ended(0));
        assert_eq!((r.offset, r.end), (7, 42));
        assert_eq!(r.data, b"hello");
        let r = parse_read("running - 0 0\n\n").unwrap();
        assert_eq!(r.status, JobStatus::Running);
        assert!(r.data.is_empty());
        assert_eq!(parse_read("gone\n").unwrap().status, JobStatus::Gone);
        assert!(parse_read("ended zero 0 0\n\n").is_err());
        assert!(parse_read("what\n").is_err());
    }

    #[test]
    fn a_list_line_parses() {
        let j =
            parse_list_line("0123456789abcdef running - 3 background 17 1700000000 0 npm run dev")
                .unwrap();
        assert_eq!(j.status, JobStatus::Running);
        assert!(j.kept);
        assert_eq!(j.num, Some(3));
        assert_eq!(j.kind.as_deref(), Some("background"));
        assert_eq!((j.seen, j.started), (17, 1_700_000_000));
        assert_eq!(j.label, "npm run dev");
        let held = parse_list_line("00ff exited 1 - - 0 5 0 ").unwrap();
        assert!(!held.kept);
        assert_eq!(held.status, JobStatus::ShellExited(1));
        assert_eq!(held.num, None);
        // Started kept, before its number was recorded.
        let kept = parse_list_line("00ff running - - background 0 5 0 x").unwrap();
        let ended = parse_list_line("00ff ended 2 4 adopted 9 5 8 x y").unwrap();
        assert_eq!(
            (ended.status, ended.ended, ended.label.as_str()),
            (JobStatus::Ended(2), 8, "x y")
        );
        assert!(kept.kept && kept.num.is_none());
        assert!(parse_list_line("../x running - - - 0 0 0 x").is_none());
    }

    #[test]
    fn survivors_are_described_for_the_model() {
        assert_eq!(describe_survivors("12"), "process 12 in the sandbox");
        assert_eq!(
            describe_survivors("12,34"),
            "processes 12, 34 in the sandbox"
        );
        assert_eq!(describe_survivors(""), "a process in the sandbox");
    }

    #[test]
    fn labels_and_kinds_stay_one_field() {
        assert_eq!(one_line("a\nb\tc"), "a b c");
        assert_eq!(one_word("back ground"), "background");
        assert_eq!(one_word(""), "-");
    }
}
