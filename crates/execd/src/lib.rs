//! execd: an exec endpoint that runs inside a sandbox and speaks the agent's exec protocol.
//!
//! The normative spec is the module doc of `crates/agent/src/exec_endpoint.rs`. execd implements
//! all of it: v1.1 (one request, one JSON answer, stdin), v1.2 streaming (frames, heartbeats,
//! disconnect kills, leftover processes), v1.3 sandbox jobs and v1.4 job-end notify. See this
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
use std::sync::atomic::AtomicU64;
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio_util::sync::CancellationToken;

mod exec;
mod frame;
mod http;
mod jobs;
mod notify;
mod proc;

pub use notify::{
    ATTEMPT_TIMEOUT, CallbackAttempt, CallbackKind, NOTIFY_VERSION, NotifyTarget, parse_notify,
};
pub use proc::{become_subreaper, reap_once, reap_orphans};

/// The spec's heartbeat: a stream silent for this long gets `{}` (v1.2).
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

/// How much of a job's output is kept: the spec's floor (v1.3), 8 MiB. Older bytes read back as a
/// gap.
pub const JOB_OUTPUT_RETENTION: usize = 8 * 1024 * 1024;

/// The largest answer the agent accepts (`DEFAULT_MAX_RESPONSE_BYTES`, 16 MiB). A v1.1 answer is
/// one JSON object; the agent refuses a bigger one, so output past this would be thrown away at the
/// far end anyway.
pub const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// The largest request the agent sends, in bytes. Derived, not chosen: the biggest thing the agent
/// ever puts in a request is a file's content as `stdin_base64` (`ShellFs::write_slice`), and the
/// biggest file it writes is an edit's, capped at `MAX_EDIT_BYTES` = 64 MiB. Base64 turns 64 MiB
/// into `ceil(64 MiB / 3) * 4` = 89,478,488 bytes. The rest of the envelope is the command and its
/// arguments, which the kernel itself refuses past `ARG_MAX` (2 MiB on a default 8 MiB stack), so
/// the JSON that carries them, even at its worst, fits in that much again.
pub const MAX_REQUEST_BYTES: usize = (64 * 1024 * 1024usize).div_ceil(3) * 4 + 2 * 1024 * 1024;

/// What the deployment tells execd. [`Options::new`] is a conforming v1.4 endpoint with the spec's
/// numbers and notify off, as an operator's deployment setting starts out.
#[derive(Clone)]
pub struct Options {
    /// Where a command runs when its request names no `cwd`.
    pub root: PathBuf,
    /// `HOME` for every command; `None` inherits execd's own.
    pub home: Option<PathBuf>,
    /// Answer `401` to any request without `Authorization: Bearer <this>` (`/livez` excepted).
    /// `None` accepts everything.
    pub bearer: Option<String>,
    /// A stream silent this long gets a heartbeat. The spec's is [`HEARTBEAT_INTERVAL`]; a test
    /// that wants to see one quickly shortens it.
    pub heartbeat: Duration,
    /// Offer sandbox jobs natively (v1.3). Off, the `job` field is an unknown field and the agent
    /// runs jobs from plain exec commands (the baseline).
    pub native_jobs: bool,
    /// The deployment says a kept job survives its sandbox going idle. v1.4 notify is advertised
    /// only then (N1), and like the spec's setting it is off unless the operator turns it on.
    pub kept_jobs_survive_idle: bool,
    /// How much of each job's output is kept; see [`JOB_OUTPUT_RETENTION`].
    pub job_output_retention: usize,
    /// The most output a v1.1 answer carries; see [`MAX_RESPONSE_BYTES`].
    pub max_response_bytes: usize,
    /// The most a request may carry; see [`MAX_REQUEST_BYTES`].
    pub max_request_bytes: usize,
    /// Told of every request, stream and callback attempt. `None` costs one branch per event.
    pub observer: Option<Arc<dyn Observer>>,
}

impl Options {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            home: None,
            bearer: None,
            heartbeat: HEARTBEAT_INTERVAL,
            native_jobs: true,
            kept_jobs_survive_idle: false,
            job_output_retention: JOB_OUTPUT_RETENTION,
            max_response_bytes: MAX_RESPONSE_BYTES,
            max_request_bytes: MAX_REQUEST_BYTES,
            observer: None,
        }
    }

    /// Whether the job probe offers notify (natively, or for the baseline).
    pub fn advertises_notify(&self) -> bool {
        self.kept_jobs_survive_idle
    }
}

impl std::fmt::Debug for Options {
    /// Never shows the bearer.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Options")
            .field("root", &self.root)
            .field("home", &self.home)
            .field("bearer", &self.bearer.as_ref().map(|_| "<redacted>"))
            .field("heartbeat", &self.heartbeat)
            .field("native_jobs", &self.native_jobs)
            .field("kept_jobs_survive_idle", &self.kept_jobs_survive_idle)
            .field("job_output_retention", &self.job_output_retention)
            .field("max_response_bytes", &self.max_response_bytes)
            .field("max_request_bytes", &self.max_request_bytes)
            .field("observer", &self.observer.is_some())
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
    /// A v1.4 callback attempt ended.
    fn callback(&self, _attempt: &CallbackAttempt) {}
}

/// Everything a request handler shares.
pub(crate) struct Ctx {
    pub opts: Options,
    pub jobs: jobs::Jobs,
    pub groups: proc::Groups,
    /// The callback client, built on the first attempt: a deployment that never keeps a job with
    /// `notify` never pays for it.
    pub client: tokio::sync::OnceCell<Result<reqwest::Client, String>>,
    /// The kernel's clock starts here.
    pub epoch: Instant,
    pub next_seq: AtomicU64,
}

/// A running server.
pub struct Handle {
    addr: SocketAddr,
    stop: CancellationToken,
    accept: tokio::task::JoinHandle<()>,
    ctx: Arc<Ctx>,
}

impl Handle {
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Stop accepting, drop every open connection, kill every process group still running (the
    /// commands and the jobs), and return how many groups that was.
    pub async fn shutdown(self) -> usize {
        self.stop.cancel();
        let _ = self.accept.await;
        self.ctx.groups.kill_all()
    }
}

/// Serve on `listener`, which the server holds until [`Handle::shutdown`]. Must be called inside a
/// tokio runtime.
pub fn serve(listener: tokio::net::TcpListener, opts: Options) -> std::io::Result<Handle> {
    let addr = listener.local_addr()?;
    let ctx = Arc::new(Ctx {
        opts,
        jobs: jobs::Jobs::default(),
        groups: proc::Groups::default(),
        client: tokio::sync::OnceCell::new(),
        epoch: Instant::now(),
        next_seq: AtomicU64::new(0),
    });
    let stop = CancellationToken::new();
    let accept = tokio::spawn(http::accept_loop(listener, ctx.clone(), stop.clone()));
    Ok(Handle {
        addr,
        stop,
        accept,
        ctx,
    })
}
