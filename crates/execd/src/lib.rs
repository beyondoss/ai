//! execd: an exec endpoint that runs inside a sandbox and speaks the agent's exec protocol.
//!
//! The normative spec is the module doc of `crates/sandbox/src/exec_endpoint.rs`. execd implements
//! all of it: one request and one JSON answer (stdin in, base64 bytes out), streaming (frames, heartbeats,
//! disconnect kills, leftover processes), sandbox jobs and job-end notify. See this
//! crate's `ARCHITECTURE.md` for the request flow, the process model and every bound.
//!
//! This is the library the `execd` binary is a thin wrapper around, so a test harness can embed the
//! very code that ships:
//!
//! ```no_run
//! # async fn demo() -> std::io::Result<()> {
//! let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
//! let server = execd::serve(listener, execd::Options::new("/work"))?;
//! println!("exec endpoint at http://{}/", server.local_addr());
//! server.shutdown().await;
//! # Ok(())
//! # }
//! ```

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio_util::sync::CancellationToken;

mod admit;
mod exec;
mod fault;
mod frame;
mod http;
/// The job table's decisions, proven in Verus (`crates/execd-jobs`); the model checks the function
/// production calls.
#[doc(hidden)]
pub use execd_jobs as job_kernel;
/// The heartbeat, the one-shot's first answer and a job's signal scope, proven in Verus
/// (`crates/execd-wire`); `exec.rs` and `proc.rs` call it for each.
#[doc(hidden)]
pub use execd_wire as wire_kernel;
mod jobs;
mod ledger;
pub mod log;
mod notify;
mod proc;
mod ring;
mod runas;
mod spool;
mod tail;
mod term;

pub use notify::{
    ATTEMPT_TIMEOUT, CallbackAttempt, CallbackKind, NOTIFY_VERSION, NotifyTarget, parse_notify,
};
/// Entry points for the fuzz targets and the parser tests; not part of the API.
#[doc(hidden)]
pub mod __fuzz {
    /// Parse a request body as the server does after the bearer: `true` if it is accepted, `false`
    /// if the server would answer `400`. Never panics.
    pub fn request(bytes: &[u8]) -> bool {
        crate::http::fuzz_request(bytes)
    }

    /// Hold the job output ring to a reference buffer over the operations `bytes` describe.
    pub fn ring(bytes: &[u8]) {
        crate::ring::fuzz_ring(bytes);
    }
}

pub use hyper::HeaderMap;
pub use proc::become_subreaper;
pub use runas::{HELPER_FLAG, Resolved, RunAs, helper_main};
pub use spool::remove_stale_jobs_dirs;

/// How long a stop (`SIGTERM`) gives a job's processes to leave after their own `SIGTERM` before
/// `SIGKILL` (the `grace_ms` default of a job `kill`).
pub const JOB_STOP_GRACE: Duration = Duration::from_millis(300);

/// The spec's heartbeat: a stream silent for this long gets `{}` (streaming).
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_millis(execd_wire::HEARTBEAT_MS);

/// How much of a job's output is kept: the spec's floor (jobs), 8 MiB. Older bytes read back as a
/// gap.
pub const JOB_OUTPUT_RETENTION: usize = 8 * 1024 * 1024;

/// How long an ended job keeps its output before it is dropped and only its status is kept
/// ([`Retention::output_ms`]).
pub const ENDED_OUTPUT_MS: u64 = 15 * 60 * 1000;

/// The most output retained over ended jobs ([`Retention::bytes`]).
pub const ENDED_BYTES: u64 = 256 * 1024 * 1024;

/// The most ended entries a server holds ([`Retention::max`]): 4096, one job every 21 seconds for a
/// day, which is what makes [`ENDED_ENTRY_MS`] a clock a busy sandbox can rely on rather than one the
/// ceiling undercuts. Measured cost of a full table (this host, release binary): 12-15 MiB of RSS
/// against 7 MiB at 256, a 1.5 MB ledger, and 12-22 ms for the change that asked, the ledger being
/// rewritten whole off the runtime (`benches/retention_scale.py`).
pub const ENDED_MAX: usize = 4096;

/// How long an ended job's entry is kept ([`Retention::entry_ms`]): 24 hours, so an unread result has
/// a predictable lifetime ([`ENDED_MAX`] is then the backstop, not the deadline).
pub const ENDED_ENTRY_MS: u64 = 24 * 60 * 60 * 1000;

/// Default for [`Options::disk_writers`]: the blocking disk jobs (job output batches, the jobs
/// ledger, a retention sweep's unlinks) that run off the runtime at once.
pub const DISK_WRITERS: usize = 8;

/// Descriptors one request in flight holds at most (measured on the binary: a stream or a one-shot
/// 3, a job start 4: the connection's socket and two pipes or spool files).
const FDS_PER_REQUEST: u64 = 4;

/// [`Options::max_in_flight`] for a soft `RLIMIT_NOFILE` of `soft` (`None`: unlimited, so no cap):
/// what is left after an eighth of the limit (at least 64: the listener, the inotify instance, the
/// ledger, the spare, idle connections) is shared out at [`FDS_PER_REQUEST`] each, and never less
/// than 1, so a stingy limit still serves.
pub fn stream_limit(soft: Option<u64>) -> Option<usize> {
    let soft = soft?;
    let reserve = (soft / 8).max(64);
    let n = (soft.saturating_sub(reserve) / FDS_PER_REQUEST).max(1);
    Some(usize::try_from(n).unwrap_or(usize::MAX))
}

/// The largest answer the agent accepts (`DEFAULT_MAX_RESPONSE_BYTES`, 16 MiB). A one-shot answer is
/// one JSON object; the agent refuses a bigger one, so output past this would be thrown away at the
/// far end anyway.
pub const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// The largest request the agent sends, in bytes. Derived, not chosen: the biggest thing the agent
/// ever puts in a request is a file's content as `stdin` (`ShellFs::write_slice`), and the
/// biggest file it writes is an edit's, capped at `MAX_EDIT_BYTES` = 64 MiB. Base64 turns 64 MiB
/// into `ceil(64 MiB / 3) * 4` = 89,478,488 bytes. The rest of the envelope is the command and its
/// arguments, which the kernel itself refuses past `ARG_MAX` (2 MiB on a default 8 MiB stack), so
/// the JSON that carries them, even at its worst, fits in that much again.
pub const MAX_REQUEST_BYTES: usize = (64 * 1024 * 1024usize).div_ceil(3) * 4 + 2 * 1024 * 1024;

/// Decides whether a request may run, from its headers alone: see [`Options::authorizer`]. Called
/// inline on the connection's task, so it must be quick and must not block.
pub type Authorizer = Arc<dyn Fn(&HeaderMap) -> bool + Send + Sync>;

/// Answers a `job` op this server does not implement: `Some((status, body))`, or `None` for the
/// usual 400. See [`Options::job_op_hook`].
pub type JobOpHook = Arc<dyn Fn(&str, &Value) -> Option<(u16, Value)> + Send + Sync>;

/// The request a [`RouteHook`] is shown: method, path, headers and a body that has not been read.
pub type HttpRequest = hyper::Request<hyper::body::Incoming>;

/// A response body an embedder streams: any frames, sent as they are produced.
pub type RouteBody = http_body_util::combinators::BoxBody<bytes::Bytes, std::io::Error>;

/// What a claimed request is answered with: the head is sent as soon as this resolves, then the body
/// streams for as long as it produces frames.
pub type RouteAnswer =
    std::pin::Pin<Box<dyn std::future::Future<Output = hyper::Response<RouteBody>> + Send>>;

/// A [`RouteHook`]'s verdict on a request.
#[allow(
    clippy::large_enum_variant,
    reason = "boxing the request would add an allocation to every request whenever a hook is set"
)]
pub enum Route {
    /// The hook handles it: execd sends this answer and does nothing else with the request.
    Claimed(RouteAnswer),
    /// Not the hook's: execd handles the request, handed back untouched, as it would without a hook.
    Declined(HttpRequest),
}

/// Lets an embedder claim requests by method and path before execd reads them as exec requests: see
/// [`Options::route_hook`]. Called inline on the connection's task, so it must decide quickly and
/// must not block; the work belongs in the returned future.
pub type RouteHook = Arc<dyn Fn(HttpRequest) -> Route + Send + Sync>;

/// How long an ended job keeps what it holds, and what the server does about it. Two tiers, one
/// sweeper (`jobs::sweep`): the **output** is dropped first (a *tombstone*: the entry keeps its
/// offsets, `state`, code and end, the ring goes to `base == end`, and a late reader gets the status
/// plus a gap), and the **entry** is taken later, and only through the gate `remove` takes.
///
/// Every number is a knob, and every off value (`0`) is exactly what execd always did: nothing is
/// dropped and nothing is reaped, so a deployment that wants no retention sets all four to `0` and
/// keeps today's behaviour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Retention {
    /// `EXECD_JOB_OUTPUT_MS`: drop an ended job's output this long after its end, keeping its entry.
    /// `0`: never. Default [`ENDED_OUTPUT_MS`] (15 min).
    pub output_ms: u64,
    /// `EXECD_JOBS_ENDED_BYTES`: the most output retained over ended jobs. Past it the oldest is
    /// tombstoned first, so a late reader gets the same answer it would past the retention: a gap.
    /// `0`: unbounded. Default [`ENDED_BYTES`] (256 MiB).
    pub bytes: u64,
    /// `EXECD_JOBS_ENDED_MAX`: the most ended entries the table holds. Past it the oldest ended
    /// entry is reaped, with no clock involved: a flood of tiny jobs grows only entries. `0`:
    /// unbounded. Default [`ENDED_MAX`] (4096).
    pub max: usize,
    /// `EXECD_JOB_ENTRY_MS`: reap an ended job's entry, status included, this long after its end.
    /// `0`: never. Default [`ENDED_ENTRY_MS`] (24 h): an ended result's lifetime is then predictable,
    /// and a client that reconnects later than that gets the defined `404`/`gone`, never a stream.
    pub entry_ms: u64,
}

impl Default for Retention {
    fn default() -> Self {
        Self {
            output_ms: ENDED_OUTPUT_MS,
            bytes: ENDED_BYTES,
            max: ENDED_MAX,
            entry_ms: ENDED_ENTRY_MS,
        }
    }
}

impl Retention {
    /// Whether any of the four is on. With all four at `0` the sweeper is never started and no
    /// ended job is touched.
    pub fn enabled(&self) -> bool {
        self.output_ms > 0 || self.bytes > 0 || self.max > 0 || self.entry_ms > 0
    }
}

/// What the deployment tells execd. [`Options::new`] is a conforming notify endpoint with the spec's
/// numbers and notify off, as an operator's deployment setting starts out.
#[derive(Clone)]
pub struct Options {
    /// Where a command runs when its request names no `cwd`.
    pub root: PathBuf,
    /// `HOME` for every command; `None` inherits execd's own.
    pub home: Option<PathBuf>,
    /// Answer `401` to any request without `Authorization: Bearer <this>` (`/livez` excepted).
    /// `None` accepts everything, which [`serve`] allows only on a loopback listener unless
    /// [`Self::insecure_no_auth`] says otherwise.
    pub bearer: Option<String>,
    /// The embedder's own authentication, instead of [`Self::bearer`]: every request except the
    /// health paths (`/livez`, `/healthz`) is allowed if and only if this returns `true`, and is
    /// answered `401` (exactly as for a bad bearer) otherwise. Called once per request, before its
    /// body is read. It counts as authentication for [`serve`]'s fail-closed rule, so a non-loopback
    /// listener needs no bearer with it. Setting both this and [`Self::bearer`] is a configuration
    /// error: [`serve`] returns `InvalidInput`. The binary has no way to set it.
    pub authorizer: Option<Authorizer>,
    /// Allow [`serve`] with no bearer on a listener that is not loopback: anyone who can reach it
    /// runs commands. Off by default, so an embedding host cannot do it by forgetting the bearer.
    pub insecure_no_auth: bool,
    /// A stream silent this long gets a heartbeat. The spec's is [`HEARTBEAT_INTERVAL`]; a test
    /// that wants to see one quickly shortens it.
    pub heartbeat: Duration,
    /// Run every command as this uid, gid and groups instead of execd's own. The host must call
    /// [`helper_main`] first thing, and should prove the switch works with [`RunAs::check`].
    /// `None`: commands share execd's uid, and can read its memory and its environment.
    pub run_as: Option<RunAs>,
    /// How much of each job's output is kept; see [`JOB_OUTPUT_RETENTION`].
    pub job_output_retention: usize,
    /// How long an ended job keeps its output and its entry, and what the ceilings are; see
    /// [`Retention`]. All four off is what execd always did.
    pub retention: Retention,
    /// Where the retained output of kept jobs is spooled, as chunk files under a directory of
    /// execd's own (mode 0700); `None` is `$TMPDIR`, else `/tmp`. Use a disk-backed directory:
    /// on tmpfs the output is RAM.
    pub jobs_dir: Option<PathBuf>,
    /// Where the jobs ledger is kept (a directory of execd's own, mode 0700), so a restart of execd
    /// (a crash, a kill, a new binary) finds the jobs it started: the ones whose process groups run
    /// on are adopted into the table (listable, signalled, removed, and counted by
    /// [`Handle::activity`]); the ones that ended keep the status they ended with; one whose group
    /// is gone ends with its shell's saved status, else `-1`. Each entry names the job's spool
    /// directory (`ledger.rs`): its output is files it inherits (`tail.rs`), so it writes through the
    /// restart, and the new execd rebuilds its ring and answers `read` at the same offsets. A
    /// shutdown of this server leaves a job's group running when this is set, and kills it when not.
    /// `None`: nothing is kept, as before. Use a directory that dies with the machine the jobs run
    /// on (tmpfs): a ledger that outlives its processes only names groups that are gone.
    pub state_dir: Option<PathBuf>,
    /// The most output a one-shot answer carries; see [`MAX_RESPONSE_BYTES`].
    pub max_response_bytes: usize,
    /// The most a request may carry; see [`MAX_REQUEST_BYTES`].
    pub max_request_bytes: usize,
    /// Told of every request, stream and callback attempt. `None` costs one branch per event.
    pub observer: Option<Arc<dyn Observer>>,
    /// Close a connection that has had no request in flight for this long: HTTP/1.1 is closed, HTTP/2
    /// is sent a GOAWAY. A request in flight (a stream or a held job, however long, even while it
    /// only sends heartbeats) is never cut, and the clock starts again when its answer ends. `None`
    /// (the default) starts no timer at all. For a sandbox that sleeps when it has no network
    /// traffic behind a proxy that pools connections.
    pub idle_timeout: Option<Duration>,
    /// Asked about a `job` op execd does not implement, with the op's name and the whole `job`
    /// object: `Some((status, body))` is sent as a JSON response, `None` gets the usual 400. Called
    /// only for an unknown op, after authentication and request parsing, never for one execd handles.
    /// It is synchronous and runs on the server's own thread, so it must be quick and must not block.
    pub job_op_hook: Option<JobOpHook>,
    /// The binary a terminal's keeper is started from: execd's own, run in keeper mode. Default
    /// `/proc/self/exe`; a host that embeds execd in a process that does not call [`helper_main`]
    /// first (a test binary) names the `execd` binary here.
    pub helper_exe: Option<PathBuf>,
    /// At most this many open terminals; a start beyond it is `503 no pseudo-terminal is free`, as
    /// when the kernel's table is full. `None` (the default) is the kernel's own limit. Set only by
    /// `EXECD_TERMINALS_MAX`, so a test can reach the refusal without filling the machine's ptys.
    pub terminals_max: Option<usize>,
    /// At most this many requests in flight (a stream, a one-shot, a job start: everything but the
    /// health paths and a request the route hook claims); one more is answered `503` with
    /// `retry-after: 1` and holds nothing. Each holds descriptors (3 or 4 measured: its socket and its
    /// pipes or spool files), so a limit sized from `RLIMIT_NOFILE` ([`stream_limit`]) keeps the
    /// descriptors from running out under the requests execd already took. `None` (the default here,
    /// and `EXECD_MAX_STREAMS=0`) is no cap: the descriptor limit alone ends it.
    pub max_in_flight: Option<usize>,
    /// At most this many blocking disk jobs at once, each on a thread off the runtime: a job's
    /// output batch, a write of the jobs ledger, a retention sweep's unlinks; one beyond it waits its
    /// turn (a job's spool file holds what it wrote meanwhile, and a job that ended is not slowed).
    /// `0` writes on the runtime, as before this existed (`EXECD_DISK_WRITERS=0`).
    pub disk_writers: usize,
    /// Out of descriptors, answer the connections waiting in the kernel's queue `503` with a spare
    /// descriptor ([`http::accept_loop`]). `false` (`EXECD_SHED_WHEN_OUT_OF_FDS=0`) leaves them
    /// waiting, as before this existed.
    pub shed_when_out_of_fds: bool,
    /// Shown every request except the health paths, after [`Self::authorizer`] or [`Self::bearer`]
    /// has accepted it (a refused one is answered `401` and the hook never sees it) and before its
    /// body is read: [`Route::Claimed`] takes the request over, [`Route::Declined`] hands it back to
    /// execd, which then treats it exactly as it would without a hook. With a hook set, a refused
    /// request of any method gets `401`, where without one a non-`POST` gets `405` unchecked.
    ///
    /// A claimed request is not execd's work. It is not counted in [`Activity::in_flight`], nor in the
    /// connection's idle timeout, so a stream a claimed handler holds open for hours does not keep a
    /// host that pauses on [`Handle::activity`] awake: if the claimed work must keep the host up, the
    /// embedder says so itself. The answer's head goes out as soon as the future resolves, so a
    /// handler can reply at once and then stream the body while it reads the request body, full
    /// duplex over HTTP/2 (HTTP/1.1 is half duplex in practice for most clients). The request body
    /// is not bounded by [`Self::max_request_bytes`], which does not apply to a claimed request: the
    /// handler owns its own body bounds. The reference endpoint claims nothing.
    pub route_hook: Option<RouteHook>,
}

impl Options {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            home: None,
            bearer: None,
            authorizer: None,
            insecure_no_auth: false,
            heartbeat: HEARTBEAT_INTERVAL,
            run_as: None,
            job_output_retention: JOB_OUTPUT_RETENTION,
            retention: Retention::default(),
            jobs_dir: None,
            state_dir: None,
            max_response_bytes: MAX_RESPONSE_BYTES,
            max_request_bytes: MAX_REQUEST_BYTES,
            observer: None,
            idle_timeout: None,
            job_op_hook: None,
            helper_exe: None,
            terminals_max: None,
            max_in_flight: None,
            disk_writers: DISK_WRITERS,
            shed_when_out_of_fds: true,
            route_hook: None,
        }
    }
}

impl std::fmt::Debug for Options {
    /// Never shows the bearer.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Options")
            .field("root", &self.root)
            .field("home", &self.home)
            .field("bearer", &self.bearer.as_ref().map(|_| "<redacted>"))
            .field("authorizer", &self.authorizer.is_some())
            .field("insecure_no_auth", &self.insecure_no_auth)
            .field("heartbeat", &self.heartbeat)
            .field("run_as", &self.run_as)
            .field("job_output_retention", &self.job_output_retention)
            .field("retention", &self.retention)
            .field("jobs_dir", &self.jobs_dir)
            .field("state_dir", &self.state_dir)
            .field("max_response_bytes", &self.max_response_bytes)
            .field("max_request_bytes", &self.max_request_bytes)
            .field("observer", &self.observer.is_some())
            .field("idle_timeout", &self.idle_timeout)
            .field("job_op_hook", &self.job_op_hook.is_some())
            .field("route_hook", &self.route_hook.is_some())
            .finish()
    }
}

/// How a streamed command's exchange ended, as the endpoint saw it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamEnd {
    /// The command exited (or was signalled) and its final frame was sent.
    Finished,
    /// The endpoint enforced `timeout_ms`.
    TimedOut,
    /// The client went away before the final frame. `killed` is whether the endpoint then killed
    /// the process group.
    Disconnected { killed: bool },
}

/// A streamed command that has started.
#[derive(Clone, Debug)]
pub struct StreamStart {
    /// Unique per server, in the order streams start: how [`Observer::stream_ended`] names one.
    pub seq: u64,
    /// The command's process group (it leads it).
    pub pgid: u32,
    pub command: String,
    pub args: Vec<String>,
    pub at: Instant,
}

/// Told what the endpoint does. Every method has an empty default and is called inline, so an
/// implementation must be quick and must not block. Unset ([`Options::observer`]) it costs one
/// branch per event and builds none of the arguments.
pub trait Observer: Send + Sync + 'static {
    /// A request that passed authentication and parsed: its headers (names lowercased) and its
    /// JSON body.
    fn request(&self, _headers: &[(String, String)], _body: &Value) {}
    /// A streamed command (or a held job start) began.
    fn stream_started(&self, _start: &StreamStart) {}
    /// The stream `seq` ended so.
    fn stream_ended(&self, _seq: u64, _end: StreamEnd, _at: Instant) {}
    /// A notify callback attempt ended.
    fn callback(&self, _attempt: &CallbackAttempt) {}
}

/// A kept jobs job that has not ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeptJob {
    /// The job's process group.
    pub pgid: i32,
    pub scope: String,
    pub id: String,
}

/// A kept terminal that has not ended and whose keeper holds it: what a host that lets idle machines
/// sleep needs to tell a terminal waiting at its prompt from one at work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Terminal {
    pub scope: String,
    pub id: String,
    /// The session's process group.
    pub pgid: i32,
    /// The slave's device number (`st_rdev` of its `/dev/pts/N`); 0 if it could not be read.
    pub pts_rdev: u64,
    /// The `st_dev` of the devpts the slave lives on: a nested devpts reuses `st_rdev` numbers, so
    /// the two together name the slave. 0 if it could not be read.
    pub pts_fs: u64,
    /// A stream (`attach`, a held start) is open on it now.
    pub attached: bool,
    /// The job's processes: its group plus every marked process that left it (the set `kill` signals).
    pub members: Vec<u32>,
}

/// What the server is doing now.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Activity {
    /// Requests from handler start until the response body is dropped: an open stream, an `attach`,
    /// a held `start` and a long one-shot command all count. `/livez` and `/healthz` never do.
    pub in_flight: usize,
    /// Every job that was kept and has not ended, with its process group.
    pub running_kept: Vec<KeptJob>,
    /// Every job in the table that has not ended, in any scope, held or kept.
    pub jobs_alive: usize,
    /// Jobs whose notify notification is owed and not yet settled: an attempt in flight or a retry
    /// waiting on its backoff, from the moment a callback becomes owed until it is settled,
    /// abandoned at the attempt limit, or the job is removed. A host that pauses idle machines
    /// must stay awake while this is not zero.
    pub callbacks_owed: usize,
    /// Every kept terminal that has not ended (see [`Terminal`]).
    pub terminals: Vec<Terminal>,
}

/// Everything a request handler shares.
pub(crate) struct Ctx {
    pub opts: Options,
    /// The environment variables no command may inherit: every `EXECD_*` execd was started with.
    pub scrub: Vec<std::ffi::OsString>,
    pub jobs: jobs::Jobs,
    /// Where kept jobs' output is held.
    pub spool: spool::JobsDir,
    /// Wakes a job's tail when its spool file is written (`tail.rs`).
    pub tails: Arc<tail::Hub>,
    /// Set when a shutdown begins: a held job's client leaving with it is not a client leaving.
    pub stopping: std::sync::atomic::AtomicBool,
    /// Set with `stopping` when the shutdown is a restart (`SIGHUP`): a held job nobody kept is left
    /// for the next execd instead of killed with its connection.
    pub restarting: std::sync::atomic::AtomicBool,
    pub groups: proc::Groups,
    /// The callback client, built on the first attempt: a deployment that never keeps a job with
    /// `notify` never pays for it.
    pub client: tokio::sync::OnceCell<Result<reqwest::Client, String>>,
    /// The kernel's clock starts here.
    pub epoch: Instant,
    pub next_seq: AtomicU64,
    /// Requests in flight, server-wide: the sum of every connection's own count.
    pub in_flight: admit::InFlightCount,
    /// Permits for [`Options::disk_writers`] blocking disk jobs at once (see [`Ctx::disk_turn`]).
    pub disk_slots: tokio::sync::Semaphore,
    /// Bumped whenever [`Handle::activity`] may have changed (shared with `jobs`).
    pub activity: Arc<tokio::sync::watch::Sender<u64>>,
}

impl Ctx {
    pub fn activity_changed(&self) {
        self.activity.send_modify(|v| *v = v.wrapping_add(1));
    }

    /// A turn for one blocking disk job — a job's output batch, a ledger write, a retention sweep's
    /// unlinks — of the [`Options::disk_writers`] allowed at once, or `None` when that is `0` (the
    /// caller then does its own thing, as before the bound existed). The one gate every off-loop
    /// disk job passes, so the threads are bounded however many jobs and sweeps want one; the
    /// semaphore is closed only with the server, which never happens while a job runs.
    pub(crate) async fn disk_turn(&self) -> Option<tokio::sync::SemaphorePermit<'_>> {
        if self.opts.disk_writers == 0 {
            return None;
        }
        self.disk_slots.acquire().await.ok()
    }

    /// Write the jobs ledger for a change of a job's life, off the runtime thread and holding one of
    /// the `disk_writers` turns — the gate a job's output batch takes. The caller awaits it before it
    /// answers, so the answer still promises what it promised when the write was inline: when a
    /// `start` answers, the file names the job; when a `remove` answers, it does not. What changed is
    /// where it runs: 1.5 MB of write and rename at a full table is minutes of nothing for the
    /// requests this server exists to answer, and it no longer spends them on the runtime's thread.
    ///
    /// Ordering against another writer — a second life change, or a retention sweep's — is the
    /// ledger's: `Ledger::save` takes the file's writer lock and only then snapshots the table, so
    /// writes are serialized and the last file written is the last state snapshotted, whichever
    /// thread wrote it. A write that was asked for earlier can never land over a newer one.
    pub(crate) async fn save_jobs(self: &Arc<Self>) {
        if !self.jobs.ledger().enabled() {
            return;
        }
        if cfg!(feature = "mut_ledger_write_inline") {
            self.jobs.save();
            return;
        }
        let turn = self.disk_turn().await;
        let me = self.clone();
        proc::off_runtime(move || me.jobs.save()).await;
        drop(turn);
    }

    /// [`Self::save_jobs`] for a keyed start's claim (K10): written, synced, renamed and its directory
    /// synced on a disk-writer thread, and how that came out returned, so the start still refuses when
    /// the claim is not on disk. A claim nobody wrote is a refusal, never a spawn.
    pub(crate) async fn claim_jobs(self: &Arc<Self>) -> job_kernel::Persist {
        let turn = self.disk_turn().await;
        let me = self.clone();
        let wrote = proc::off_runtime(move || Some(me.jobs.claim()))
            .await
            .unwrap_or(job_kernel::Persist::Failed);
        drop(turn);
        wrote
    }

    /// [`Self::save_jobs`] for a shutdown that ends every job: the ledger is emptied (`SIGTERM`), so
    /// the next execd adopts nothing.
    pub(crate) async fn forget_jobs(self: &Arc<Self>) {
        let turn = self.disk_turn().await;
        let me = self.clone();
        proc::off_runtime(move || me.jobs.forget()).await;
        drop(turn);
    }
}

/// A running server.
pub struct Handle {
    addr: SocketAddr,
    stop: CancellationToken,
    accept: tokio::task::JoinHandle<()>,
    reaper: CancellationToken,
    ctx: Arc<Ctx>,
}

impl Handle {
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// What the server is doing now.
    pub fn activity(&self) -> Activity {
        let (jobs_alive, running_kept, callbacks_owed) = self.ctx.jobs.alive();
        Activity {
            in_flight: self.ctx.in_flight.get(),
            running_kept,
            jobs_alive,
            callbacks_owed,
            terminals: self.ctx.jobs.terminals(),
        }
    }

    /// Bumps whenever [`Self::activity`] may have changed: a request starts or ends, a job is kept,
    /// a kept job ends, a job is removed. A bump with no subscriber costs a store.
    pub fn activity_changed(&self) -> tokio::sync::watch::Receiver<u64> {
        self.ctx.activity.subscribe()
    }

    /// A stop (`SIGTERM`, `SIGINT`): stop accepting, drop every open connection and end every process
    /// group, a job's included, with or without a ledger (S6): jobs get `SIGTERM`, then `SIGKILL`
    /// after `JOB_STOP_GRACE`; a command's group is killed. The ledger is emptied. Returns how many
    /// groups it ended.
    pub async fn shutdown(self) -> usize {
        self.shutdown_for(job_kernel::Stop::Term).await
    }

    /// A restart (`SIGHUP`): as a stop, except that a job's group is left running and kept in the
    /// ledger when there is one (`Options::state_dir`), for the next execd to adopt (S3, S4), and
    /// killed when there is not. Returns how many groups it killed.
    pub async fn restart(self) -> usize {
        self.shutdown_for(job_kernel::Stop::Hup).await
    }

    async fn shutdown_for(self, stop: job_kernel::Stop) -> usize {
        self.ctx.stopping.store(true, Ordering::Release);
        self.ctx
            .restarting
            .store(stop == job_kernel::Stop::Hup, Ordering::Release);
        self.stop.cancel();
        let _ = self.accept.await;
        let ended = if stop == job_kernel::Stop::Term {
            jobs::stop_all(&self.ctx, JOB_STOP_GRACE).await
        } else {
            0
        };
        let killed = ended + self.ctx.groups.shutdown(stop);
        if self
            .ctx
            .groups
            .at_shutdown(job_kernel::Kind::Job, stop)
            .record
        {
            self.ctx.save_jobs().await;
        } else {
            self.ctx.forget_jobs().await;
        }
        self.ctx.spool.clean();
        self.reaper.cancel();
        killed
    }
}

/// Serve on `listener`, which the server holds until [`Handle::shutdown`]. Must be called inside a
/// tokio runtime.
pub fn serve(listener: tokio::net::TcpListener, opts: Options) -> std::io::Result<Handle> {
    let addr = listener.local_addr()?;
    if opts.bearer.is_some() && opts.authorizer.is_some() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Options::bearer and Options::authorizer are both set: choose one way to authenticate",
        ));
    }
    // Before any command can be spawned: see `proc::release_signals`.
    proc::release_signals();
    let spool = spool::JobsDir::open(opts.jobs_dir.as_deref(), opts.state_dir.as_deref())?;
    // Fail closed: no authentication is only for a listener that only this host can reach.
    if opts.bearer.is_none()
        && opts.authorizer.is_none()
        && !opts.insecure_no_auth
        && !addr.ip().to_canonical().is_loopback()
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!(
                "refusing to serve on {addr} with no bearer: anyone who can reach it could run \
                 commands (set Options::bearer or Options::authorizer, listen on loopback, or set Options::insecure_no_auth)"
            ),
        ));
    }
    let scrub = std::env::vars_os()
        .map(|(k, _)| k)
        .filter(|k| k.as_encoded_bytes().starts_with(b"EXECD_"))
        .collect();
    let activity = Arc::new(tokio::sync::watch::Sender::new(0));
    let ledger = opts
        .state_dir
        .as_deref()
        .map_or_else(ledger::Ledger::disabled, ledger::Ledger::at);
    let ledger_on = ledger.enabled();
    let disk_slots = tokio::sync::Semaphore::new(opts.disk_writers);
    let ctx = Arc::new(Ctx {
        opts,
        scrub,
        jobs: jobs::Jobs::new(activity.clone(), ledger),
        spool,
        tails: tail::Hub::new()?,
        stopping: std::sync::atomic::AtomicBool::new(false),
        restarting: std::sync::atomic::AtomicBool::new(false),
        groups: proc::Groups::new(ledger_on),
        client: tokio::sync::OnceCell::new(),
        epoch: Instant::now(),
        next_seq: AtomicU64::new(0),
        in_flight: admit::InFlightCount::new(),
        disk_slots,
        activity,
    });
    // The reaper collects every command's exit (see `proc`); it runs before the first can start.
    let reaper = CancellationToken::new();
    proc::start_reaper(reaper.clone())?;
    // The jobs a previous execd started, before the first request can name one.
    jobs::recover(&ctx);
    let stop = CancellationToken::new();
    // The retention of ended jobs (`Retention`): one task, asleep until a job ends or a deadline
    // arrives, started only when one of its four knobs is on, and stopped with the server.
    jobs::start_retention(&ctx, stop.clone());
    let accept = tokio::spawn(http::accept_loop(listener, ctx.clone(), stop.clone()));
    Ok(Handle {
        addr,
        stop,
        accept,
        reaper,
        ctx,
    })
}

#[cfg(test)]
mod stream_limit_tests {
    use super::stream_limit;

    #[test]
    fn the_cap_follows_the_descriptor_limit_and_is_never_zero() {
        assert_eq!(stream_limit(None), None);
        assert_eq!(stream_limit(Some(1024)), Some(224));
        assert_eq!(stream_limit(Some(128)), Some(16));
        assert_eq!(stream_limit(Some(8)), Some(1));
        assert_eq!(stream_limit(Some(0)), Some(1));
        assert!(stream_limit(Some(u64::MAX)).is_some());
    }
}
