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
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio_util::sync::CancellationToken;

mod exec;
mod frame;
mod http;
mod jobs;
pub mod log;
mod notify;
mod proc;
mod runas;
mod spool;

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
        crate::jobs::fuzz_ring(bytes);
    }
}

pub use hyper::HeaderMap;
pub use proc::become_subreaper;
pub use runas::{HELPER_FLAG, Resolved, RunAs, helper_main};
pub use spool::remove_stale_jobs_dirs;

/// The spec's heartbeat: a stream silent for this long gets `{}` (streaming).
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

/// How much of a job's output is kept: the spec's floor (jobs), 8 MiB. Older bytes read back as a
/// gap.
pub const JOB_OUTPUT_RETENTION: usize = 8 * 1024 * 1024;

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
    /// Where the retained output of kept jobs is spooled, as chunk files under a directory of
    /// execd's own (mode 0700); `None` is `$TMPDIR`, else `/tmp`. Use a disk-backed directory:
    /// on tmpfs the output is RAM.
    pub jobs_dir: Option<PathBuf>,
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
            jobs_dir: None,
            max_response_bytes: MAX_RESPONSE_BYTES,
            max_request_bytes: MAX_REQUEST_BYTES,
            observer: None,
            idle_timeout: None,
            job_op_hook: None,
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
            .field("jobs_dir", &self.jobs_dir)
            .field("max_response_bytes", &self.max_response_bytes)
            .field("max_request_bytes", &self.max_request_bytes)
            .field("observer", &self.observer.is_some())
            .field("idle_timeout", &self.idle_timeout)
            .field("job_op_hook", &self.job_op_hook.is_some())
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
}

/// Everything a request handler shares.
pub(crate) struct Ctx {
    pub opts: Options,
    /// The environment variables no command may inherit: every `EXECD_*` execd was started with.
    pub scrub: Vec<std::ffi::OsString>,
    pub jobs: jobs::Jobs,
    /// Where kept jobs' output is held.
    pub spool: spool::JobsDir,
    pub groups: proc::Groups,
    /// The callback client, built on the first attempt: a deployment that never keeps a job with
    /// `notify` never pays for it.
    pub client: tokio::sync::OnceCell<Result<reqwest::Client, String>>,
    /// The kernel's clock starts here.
    pub epoch: Instant,
    pub next_seq: AtomicU64,
    /// Requests in flight, server-wide: the sum of every connection's own count.
    pub in_flight: AtomicUsize,
    /// Bumped whenever [`Handle::activity`] may have changed (shared with `jobs`).
    pub activity: Arc<tokio::sync::watch::Sender<u64>>,
}

impl Ctx {
    pub fn activity_changed(&self) {
        self.activity.send_modify(|v| *v = v.wrapping_add(1));
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
            in_flight: self.ctx.in_flight.load(Ordering::Acquire),
            running_kept,
            jobs_alive,
            callbacks_owed,
        }
    }

    /// Bumps whenever [`Self::activity`] may have changed: a request starts or ends, a job starts,
    /// is kept, ends or is removed, and whenever a notify callback is owed or settled. Job output
    /// does not bump it. A bump with no subscriber costs a store.
    pub fn activity_changed(&self) -> tokio::sync::watch::Receiver<u64> {
        self.ctx.activity.subscribe()
    }

    /// Stop accepting, drop every open connection, kill every process group still running (the
    /// commands and the jobs), and return how many groups that was.
    pub async fn shutdown(self) -> usize {
        self.stop.cancel();
        let _ = self.accept.await;
        let killed = self.ctx.groups.kill_all();
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
    let spool = spool::JobsDir::open(opts.jobs_dir.as_deref())?;
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
    let ctx = Arc::new(Ctx {
        opts,
        scrub,
        jobs: jobs::Jobs::new(activity.clone()),
        spool,
        groups: proc::Groups::default(),
        client: tokio::sync::OnceCell::new(),
        epoch: Instant::now(),
        next_seq: AtomicU64::new(0),
        in_flight: AtomicUsize::new(0),
        activity,
    });
    // The reaper collects every command's exit (see `proc`); it runs before the first can start.
    let reaper = CancellationToken::new();
    proc::start_reaper(reaper.clone())?;
    let stop = CancellationToken::new();
    let accept = tokio::spawn(http::accept_loop(listener, ctx.clone(), stop.clone()));
    Ok(Handle {
        addr,
        stop,
        accept,
        reaper,
        ctx,
    })
}
