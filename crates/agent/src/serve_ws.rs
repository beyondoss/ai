//! `serve --listen <addr>` — the [`serve`](crate::serve) control protocol offered over a WebSocket
//! **or** a one-shot HTTP POST, instead of stdio. A client (the Beyond iPhone app, a script, a
//! lifecycle consumer) speaks one JSON object per WS **text** message, or POSTs that same object to
//! the same path. The protocol is **byte-identical** to stdio mode; this module is a thin transport
//! adapter over the same [`serve_session`](crate::serve::serve_session) core.
//!
//! ## The connection is a *view*, not the session's owner
//!
//! A phone driving a long agent run drops connection constantly (tunnels, locked screens). So a
//! session's lifecycle is lifted **above** any single connection: the supervisor owns a
//! `session id → running session` map, each session runs its own [`serve_session`] task, and a
//! dropped socket does **not** stop the run. The trick that makes this cheap (see [`serve_session`]'s
//! own doc comment): a session's input is an `mpsc` channel whose `Sender` is retained *here* in the
//! [`SessionHandle`], not by the socket — so a dropped connection is not an EOF, the run keeps going,
//! and a reconnecting client re-attaches to the same live task. It catches up on anything committed
//! while it was gone via the existing `get_messages {since}` command; live streaming resumes from the
//! current point. No protocol change, no frame buffering.
//!
//! ## Routing, ids, and persistence
//!
//! A connection names its session in the URL: `…/_beyond/agent?session_id=<id>` (absent ⇒ a fresh id
//! is minted; a WebSocket client learns it from any `response`/`get_state` frame, an HTTP POST client
//! from the `X-Session-Id` response header). That id is both the
//! supervisor's routing key **and** the persisted session id: it's handed to the session as
//! [`ServeConfig::session_id`], which *addresses* it in the repo — open that session, or create it under
//! exactly that id ([`crate::session_store::SessionRepo::open_or_create_id`]). So the id is stable
//! across reconnects, and a cold reconnect after a full process restart reopens the same conversation
//! rather than a blank one. `--no-session-persistence` opts out into in-memory-only sessions, which
//! still live re-attach for the process's lifetime.
//!
//! ## One task per id, ever: `Starting → Live → Stopping`
//!
//! Because the id names a file, two session tasks on one id would be two writers on one append-only
//! transcript. So an id's map entry lives exactly as long as its task: inserted at spawn (`Starting`,
//! then `Live` once the task runs its body), marked `Stopping` by the idle reaper or a graceful
//! shutdown (the retained input is dropped, so the session persists and exits), and removed by the
//! task itself as it exits — only its own incarnation's entry, never a newer one's. A reconnect that
//! finds its id `Stopping` **waits** for that exit and then spawns afresh; it never starts a rival
//! beside a task still persisting. See [`Phase`].
//!
//! ## HTTP POST
//!
//! The same listener accepts `POST /_beyond/agent?session_id=<id>` with a JSON command body — the
//! identical `{type, …}` object a WebSocket text message carries. This is how a consumer that does
//! not want to hold a socket (a job runner, a lifecycle collector, `curl`) **starts a run**: the
//! session is a view, not owned by the request, so the POST can return the moment the command is
//! accepted and the run keeps going. Combined with [`crate::lifecycle`]'s outbound POSTs, the
//! control plane is HTTP in both directions.
//!
//! - A `prompt` that is acknowledged (`{type:"ack"}`) returns **202** with that ack as the body.
//!   Events and the terminal `response` do **not** stream on this connection — attach a WebSocket
//!   (same `?session_id=`) or watch lifecycle. A `prompt` rejected before the ack (busy, missing
//!   `message`, bad `output_schema`) returns **200** with the `response` frame, same as any other
//!   command.
//! - Every other command waits for its `response` frame and returns **200**.
//! - `list_daemon_sessions` is answered here (the supervisor sees every session) and does not spawn
//!   one.
//! - Catch-up is a WebSocket-attach concern; a POST sink is not seeded with history.
//!
//! The body is the protocol frame, byte-identical to the WebSocket. The session id is an HTTP
//! header (`X-Session-Id`) because an `ack` frame does not carry it. Auth is still the front
//! door's: this crate never parses a user token.
//!
//! Because the id is a routing key, `new_session` on a live connection **keeps** it: the conversation is
//! archived into a session of its own and this one is blanked in place, so the address a client holds
//! never goes stale (see [`crate::serve::Persistence::new_session`]). Creating a genuinely new session
//! is a routing operation — connect with a new `?session_id=`.
//!
//! One limit worth naming: `switch_session` (and `fork`/`clone`) move *this process's* view to another
//! session while the routing key stays put. That's fine while the session is live, but it isn't durable
//! — if the session is reaped and later respawned, the key re-opens the session it was named for. For a
//! durable move, reconnect at `?session_id=<target>` instead.
//!
//! ## Auth
//!
//! By default there is none here: the agent authenticates no caller. Bind **loopback/internal only**
//! and trust the front door (the edge, in another repo) to have validated the client before forwarding
//! the upgrade. This module never sees or parses a user token.
//!
//! `serve --service` is the exception, and the reason is multi-tenancy rather than a change of heart:
//! one replica serving many tenants cannot infer from a loopback socket *whose* session a connection
//! is for. So every connection carries a [session grant](crate::grant) in `x-beyond-grant`, verified
//! here — before the method branch, so WebSocket and POST cannot diverge — and turned into the
//! [`ServiceSession`](crate::service::ServiceSession) that fixes that session's tenant, storage,
//! sandbox and credentials. The refusals are HTTP statuses (401/400/403/421), answered *before* the
//! upgrade; see `crate::service` and ARCHITECTURE.md's "Service mode".

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures::future::BoxFuture;
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{Duration, MissedTickBehavior};
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::handshake::server::create_response;
use tokio_tungstenite::tungstenite::protocol::{Role, WebSocketConfig};
use tokio_tungstenite::tungstenite::{Message, Utf8Bytes};
use tokio_util::sync::CancellationToken;

use crate::serve::{
    OutFanout, OutFrame, OutSink, ServeConfig, SharedOutConn, Signal, UpstreamHttp2, frame_to_line,
    lock_ignoring_poison, serve_session,
};
use crate::service::{Refusal, ServiceSession, Shards};
use crate::session_store::{
    Layout, SessionLock, acquire_session_lock, is_valid_session_id, new_id,
};

/// The fixed URL path a WebSocket upgrade must target. The front door maps a service subdomain to this
/// path on the loopback listener; any other path is rejected at the handshake.
const WS_PATH: &str = "/_beyond/agent";

/// Liveness: is this process serving at all? Answered unconditionally, so a hung shard cannot get the
/// replica killed and restarted into the same hung shard.
const LIVEZ_PATH: &str = "/livez";

/// Readiness: should this replica be sent work right now?
const READYZ_PATH: &str = "/readyz";

/// How long a `/readyz` answer is reused. A readiness probe arrives on a fixed cadence from every
/// orchestrator and load balancer watching the replica, and the check behind it is filesystem I/O on a
/// network mount — so memoize it. Two seconds is far inside any probe period, so the answer a caller
/// gets is at most one period stale, and a burst of probes (or a client looping on a 503) costs one
/// round-trip rather than one each.
const READY_CACHE_TTL: Duration = Duration::from_secs(2);

/// How long a `/readyz` caller waits for the shard probe before answering "not ready" on its own.
///
/// A healthy mount answers a `stat` plus a create/unlink in single-digit milliseconds, so this is
/// ~100× headroom; a mount that has not answered in a second is not one this replica should be given
/// a session on. It is also comfortably inside the probe's own deadline (`Dockerfile.agent`'s
/// `HEALTHCHECK --timeout=2s`), so the orchestrator reads a 503 rather than a timeout — the
/// difference between "this replica says it isn't ready" and "this replica said nothing".
const READY_PROBE_TIMEOUT: Duration = Duration::from_secs(1);

/// How long taking a session's advisory lock may take before the connection is answered 503.
///
/// The lock lives on the same network filesystem as the session, so acquiring it is exactly the I/O
/// that stops answering when a mount goes away — and without a bound the HTTP connection that asked
/// for the session simply stays open forever. Ten seconds is far longer than any healthy acquisition
/// (a `create_dir_all` plus an `open` and a `flock`) and short enough that a client learns to retry
/// while its request is still relevant. The refusal is a 503 carrying `Retry-After`, the same answer
/// every other transient session refusal gives.
const SESSION_LOCK_TIMEOUT: Duration = Duration::from_secs(10);

/// How often the server sends an unsolicited `Ping` so an idle mobile connection isn't reaped by
/// NAT/proxies. Also the granularity at which a wholly-dead socket is noticed (the ping send fails).
const PING_INTERVAL: Duration = Duration::from_secs(30);

/// Depth of a single connection's outbound frame buffer. Bounds memory per attached socket: if a
/// client's write half stalls under TCP backpressure without erroring, the session keeps broadcasting
/// streamed frames into this channel — cap it so a wedged connection can't grow memory without limit
/// (`OutFanout::broadcast` prunes the sink when it fills, dropping the dead connection). Sized to
/// absorb a normal burst of streamed event frames while keeping the worst-case per-connection buffer
/// to a few MB.
pub(crate) const OUT_CHANNEL_BOUND: usize = 1024;

/// Depth of a single connection's control-frame queue (the `Pong`s the read loop owes the client for
/// its `Ping`s). Its drain is the send task, which stalls for as long as the socket's write half is
/// backed up — so a client that floods `Ping`s while never reading would grow an unbounded queue of
/// them. Bounded, and **dropped on full**: a lost `Pong` costs nothing (the client's next `Ping`
/// re-asks, and liveness in the other direction is our own keepalive `Ping`, whose send failure is what
/// actually detects a dead socket).
const CTRL_CHANNEL_BOUND: usize = 8;

/// Cap on a single inbound WebSocket message (tungstenite's own defaults are 64 MiB per message / 16 MiB
/// per frame — far past anything this protocol needs). Every inbound message is one command line that
/// gets queued for the session, so this is what bounds the cost of any one queued command; a client that
/// exceeds it gets a protocol error and its connection closed, never a silently truncated command.
/// Generous enough for a `prompt` carrying a large pasted body.
const MAX_INBOUND_MESSAGE_BYTES: usize = 8 * 1024 * 1024;

/// Cap on HTTP/1.1 request headers (the POST path reads these itself; the WebSocket upgrade does too
/// once we peek the request-line). Far above any legitimate `Host` + `Content-Length` + a handful of
/// forwarding headers; a client that exceeds it is probing, not sending a command.
const MAX_HEADER_BYTES: usize = 16 * 1024;

/// How long an HTTP POST waits for the command's `ack`/`response` before giving up. A freshly spawned
/// session still has to open persistence and discover skills before it reads the first command, so
/// this is not a tight "the loop is idle" bound — it has to cover that startup. A `prompt`'s ack is
/// emitted the moment the turn is queued, so a healthy session replies well inside this; hitting it
/// means the session never answered, not that the run is slow.
const HTTP_REPLY_TIMEOUT: Duration = Duration::from_secs(30);

/// HTTP header carrying the session id on every POST response. An `ack` frame does not include it
/// (WebSocket clients learn it from `get_state`/`ready`); POST clients that omitted `?session_id=`
/// have no other way to address the session they just created.
const SESSION_ID_HEADER: &str = "X-Session-Id";

/// How long a **detached** session (no attached connection) stays live before the reaper reclaims it,
/// when the operator gave no `--session-idle-timeout`. The default has to be finite: without a reaper
/// the session map only grows — a connection that omits `?session_id=` mints a fresh id, and every id
/// owns a runtime task, an `Agent`, and (unless the HTTP pool is shared) a gateway client until the
/// daemon stops. It also has to be *long*, because a detached session is not a dead one: re-attaching
/// to a still-running run is the entire point
/// of the design (see the module doc), so the window must comfortably outlast a tunnel, a locked screen,
/// or a lunch break. An hour is both. Nothing is lost when it fires — a reaped session persisted on its
/// way out, and reconnecting to its id respawns it and replays from disk.
const DEFAULT_SESSION_IDLE_TIMEOUT: Duration = Duration::from_secs(60 * 60);

/// The same window in **service mode**, where an hour is not conservative but harmful.
///
/// The reasoning above is a single-user daemon's: re-attaching to a still-running run means
/// reconnecting to *this process*, so the window has to outlast a tunnel or a lunch break. On a
/// replica that is not what re-attaching means. The session lives on shared storage, so any replica
/// respawns it from disk and replays — and a session mid-run is never reaped at all
/// ([`is_reapable`] requires `!running`), so what the window actually governs is how long a
/// **detached, idle** session keeps its lock.
///
/// Holding that lock for an hour is how a failed-over session gets stranded. After a failover the
/// session is live on the replica that took it, which is not the one the edge's hash chooses; the
/// hash target answers 503 on every attempt until this window expires and the lock frees. The fleet
/// simulator reproduced exactly that: sessions intact, replayable, and unreachable from where the
/// edge looks — for as long as this constant says.
///
/// A minute instead. Long enough to ride out a page reload or a brief network blip without paying
/// for a respawn; short enough that a stranded session frees in a minute rather than an hour. Being
/// wrong in the short direction costs one read of the transcript from shared storage, which is the
/// cheap direction: being wrong in the long direction costs availability.
const DEFAULT_SERVICE_SESSION_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

/// How often a drain re-checks whether any run is still in flight.
///
/// Short enough that a drain of idle sessions is over in well under a second, long enough that the
/// poll never contends meaningfully for the table mutex the sessions themselves need.
const DRAIN_POLL: Duration = Duration::from_millis(100);

/// How long a stopped session task gets to persist and exit before whoever is waiting on it stops
/// waiting: graceful shutdown, for the whole batch; a reconnect, for its id's previous incarnation.
const JOIN_GRACE: Duration = Duration::from_secs(10);

/// Where an id's session task is in its life: `Starting → Live → Stopping → (removed)`, one way only.
///
/// The entry stays in the map for its task's **whole** life — inserted when the task is spawned,
/// removed by that task as it exits ([`ExitGuard`]) — so "the id has an entry" means exactly "a task
/// owns this id's session file". It used to mean only "attachable": a reap removed the entry first and
/// the task persisted and exited afterwards, so a reconnect in that window found nothing and spawned a
/// second writer on a file the first was still appending to.
///
/// The retained input sender lives inside the attachable phases, so a stopping session structurally
/// has none to hand out.
enum Phase {
    /// Spawned, not yet running its body. Attachable: a command queues on the input channel exactly as
    /// it does while `serve_session` opens persistence. This is the slot for start-up work that must
    /// finish before the session touches storage — in service mode, taking the session's advisory
    /// lock. The task does that work itself, awaited, with the map lock **not** held; the `Starting`
    /// entry is what keeps a concurrent pin from spawning a rival meanwhile, and the pin that spawned
    /// it waits on the outcome so "held on another replica" is a real HTTP status rather than a
    /// session that comes up and immediately dies ([`ExitGuard::go_live`]).
    Starting(mpsc::Sender<String>),
    /// Running its body ([`serve_session`]).
    Live(mpsc::Sender<String>),
    /// Told to exit and not yet gone: reaped, shut down, or its loop already ended on its own. The
    /// retained input is dropped, so the session observes EOF, persists, and exits. Not attachable — a
    /// pin waits for [`SessionHandle::exited`] and then spawns a fresh incarnation.
    Stopping,
}

impl Phase {
    /// The session's input, while it's attachable.
    fn input(&self) -> Option<&mpsc::Sender<String>> {
        match self {
            Phase::Starting(tx) | Phase::Live(tx) => Some(tx),
            Phase::Stopping => None,
        }
    }
}

/// One id's session task, reachable across connections. The input sender retained in its [`Phase`]
/// is what makes a dropped socket *not* an EOF — the session's `input_rx.recv()` pends until the next
/// command instead of shutting down (see the module doc).
struct SessionHandle {
    /// Which spawn of this id this is. Whatever acts on the entry later — the task removing it on exit,
    /// a connection unpinning — carries the incarnation it was handed, so a stale actor can never touch
    /// a newer session under the same id.
    incarnation: u64,
    /// Every attached connection feeds the one input channel held here (its `mpsc` is multi-sender), so
    /// any device can drive the session.
    phase: Phase,
    /// Whose session this is, in service mode. A session id is an address, and an address alone must
    /// never be enough: a grant for tenant B naming tenant A's live id is refused (403) at the slot
    /// rather than being allowed to attach to — and drive — a task running under A's sandbox and
    /// credentials. `None` outside service mode, where there is one tenant by construction.
    tenant: Option<String>,
    /// The session's set of attached connections — its output is **broadcast** to all of them, so a
    /// phone and a TUI (or any N of the user's devices) on one session all see the live stream at once
    /// (see [`crate::serve::OutFanout`]). Each connection registers its sink on attach and removes it on
    /// disconnect.
    out_conn: SharedOutConn,
    /// A latch, never a request: cancelled once the task has exited **and** removed this entry. What a
    /// pin on a `Stopping` id, and a graceful shutdown, wait on.
    exited: CancellationToken,
    /// How many connections are currently attached. The idle reaper only considers a session for
    /// reclamation when this reaches `0` (see [`Self::last_detached_at`]).
    attached: usize,
    /// When the last connection detached (`attached` reached `0`), for the idle reaper's clock. `None`
    /// while any connection is attached, or on a fresh session — either way it isn't reap-eligible yet.
    last_detached_at: Option<Instant>,
    /// `true` exactly while the session's [`serve_session`] loop is running a `prompt`. The reaper reads
    /// it so a detached-but-mid-run background session is never reaped out from under an in-flight turn.
    running: Arc<AtomicBool>,
    /// Why this session ended, shared with its [`ExitGuard`]. Written by whoever stops it, read once
    /// as the task exits — so the reason is recorded by the code that knows it, rather than guessed
    /// at the point of exit where every ending looks alike.
    end_reason: Arc<Mutex<crate::metrics::SessionEnd>>,
}

impl SessionHandle {
    /// → `Stopping`: drop the retained input so the session observes EOF (once no connection holds a
    /// clone either), persists, and exits. Idempotent. The entry stays until the task has gone.
    fn stop(&mut self, reason: crate::metrics::SessionEnd) {
        // First writer wins: a session the reaper already claimed that then meets a shutdown ended
        // because it was idle, not because of the deploy. Idempotent, like the phase change itself.
        {
            let mut slot = lock_ignoring_poison(&self.end_reason);
            if *slot == crate::metrics::SessionEnd::Client {
                *slot = reason;
            }
        }
        self.phase = Phase::Stopping;
    }
}

/// What this replica will still accept.
///
/// Three states, not two, and the middle one is the whole point of a drain: a replica that has been
/// told to go away must stop taking **new** sessions while still serving the ones it already holds.
/// Collapsing `Draining` into `Closed` is what makes a rolling deploy cut live conversations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Lifecycle {
    /// Serving normally.
    #[default]
    Open,
    /// SIGTERM arrived. New sessions are refused (503, so the edge retries elsewhere), `/readyz`
    /// answers 503 so the load balancer stops choosing this replica, and `/livez` stays 200 so the
    /// orchestrator does not kill it mid-drain. **Reconnects to sessions this replica still owns are
    /// accepted** — those clients have nowhere else to go until this replica lets the session's lock
    /// go, and refusing them is precisely the outage a drain exists to avoid.
    Draining,
    /// Sessions are being stopped. Nothing attaches.
    Closed,
}

/// The supervisor's `session id → session task` map, with the state that must change atomically
/// alongside it.
#[derive(Default)]
struct Table {
    sessions: HashMap<String, SessionHandle>,
    /// Stamped on the next spawn (see [`SessionHandle::incarnation`]).
    next_incarnation: u64,
    /// Set once, by graceful shutdown: from then on nothing is spawned or attached, so the sessions
    /// shutdown is waiting on are the last ones.
    lifecycle: Lifecycle,
}

/// What a session task runs once it goes live, given its id, input, output fan-out, and `running`
/// flag: [`serve_session`] in production ([`serve_session_body`]). A seam, so a test can substitute a
/// body it controls — deciding exactly when an exiting session finishes — with no gateway behind it.
///
/// The `Option<Arc<ServiceSession>>` is the **spawning** connection's verified grant: it fixes this
/// session's tenant, storage, sandbox and credentials for the task's life. A later attach still
/// presents its own grant (checked against the slot's tenant), but does not re-point any of that.
type SessionBody = Box<
    dyn Fn(
            &str,
            Option<Arc<ServiceSession>>,
            mpsc::Receiver<String>,
            SharedOutConn,
            Arc<AtomicBool>,
        ) -> BoxFuture<'static, ()>
        + Send
        + Sync,
>;

/// Owns the `session id → session task` table and what each task runs.
struct Supervisor {
    table: Arc<Mutex<Table>>,
    /// `--session-dir`, for `list_daemon_sessions`' on-disk half. `None` in service mode, where a
    /// listing is per-tenant and comes from [`Self::service`]'s shards instead.
    session_dir: Option<String>,
    /// Set by `serve --service`: this daemon authenticates every connection. `None` is the
    /// single-tenant daemon, unchanged — no grant is read and no tenant exists.
    service: Option<ServiceSupervisor>,
    /// `--metrics-listen`'s instruments, or `None` when no scrape endpoint was asked for. Held here
    /// rather than passed down because the supervisor is what every refusal and every session
    /// transition already flows through.
    metrics: Option<Arc<crate::metrics::Metrics>>,
    body: SessionBody,
}

/// What service mode needs at the *supervisor* level, as opposed to per session: the keyring every
/// connection is verified against, the mounts a tenant's sessions can live on, and how many sessions
/// this replica will hold at once.
struct ServiceSupervisor {
    verifier: Arc<crate::grant::GrantVerifier>,
    shards: Arc<Shards>,
    /// `--max-live-sessions`, or `None` when the operator passed `0` to turn the cap off. A live
    /// session costs two open descriptors (its newest segment and its lock), and a network
    /// filesystem caps both open files and locks per instance — so the default is a guard against
    /// hitting that ceiling as an unexplained I/O error deep inside a tenant's turn.
    max_live_sessions: Option<usize>,
    /// The memoized `/readyz` shard probe. Only service mode has one: without `--shard` there is
    /// nothing readiness can check beyond the listener, which is already proven by the request.
    ready: ReadyCache,
}

/// Derive a per-session config from the daemon's: address the session by its routing key and drop
/// `listen`.
///
/// Pinning `session_id` is the whole mechanism — repo mode opens exactly that session or creates it
/// under exactly that id, so the persisted id always equals the routing key. This used to rewrite
/// each session into single-file mode at `<session-dir>/<id>.jsonl` instead, purely to dodge repo
/// mode's old behavior of resolving by `cwd` and collapsing every session in a directory onto one.
/// With an id now taking precedence over the cwd match that workaround is unnecessary, and dropping
/// it fixes what it cost: daemon files were named `<id>.jsonl` where the repo names its own
/// `<created_at>_<id>.jsonl`, so `find_path`'s `_<id>.jsonl` lookup couldn't see them — a daemon
/// session appeared in `list_sessions` but `switch_session` reported it missing.
fn session_cfg(base: &ServeConfig, id: &str, service: Option<Arc<ServiceSession>>) -> ServeConfig {
    let mut c = base.clone();
    // This session's verified grant — its tenant, storage, sandbox and gateway credential. The
    // daemon's own config never carries one (at startup there is no connection), so this is the only
    // place it is set, and `Persistence::open` refuses to run in service mode without it.
    c.service = service;
    c.listen = None;
    // A spawned session must never itself re-bind a transport listener — it's driven purely
    // through its `input_rx`/`out_conn` channels by the supervisor. For the same reason it must not
    // act on OS signals: the supervisor decides when this session stops, and a session that cancels
    // its own run on SIGTERM cancels the very work a drain is counting down its grace to protect.
    c.supervised = true;
    c.listen_uds = None;
    c.listen_uds_mode = None;
    c.session_id = Some(id.to_string());
    // An addressed session selects itself; `--continue`'s "most recent for this cwd" would only be
    // able to disagree with the id the client actually routed on.
    c.continue_session = false;
    // Repo mode, always: one file can't hold the many sessions a daemon serves, so a `--session-file`
    // meant for the stdio path can't carry over. `session_dir` (or, unset, the default per-cwd repo)
    // is where they go; `--no-session-persistence` is still honored and keeps them in memory.
    c.session_file = None;
    c
}

/// The production [`SessionBody`]: run [`serve_session`] on the session's derived config.
///
/// `serve_session` is `Send` (its event sink is `FnMut + Send`, and the error type is
/// `Box<dyn Error + Send + Sync>`), so the session is a task on this process-wide runtime rather than a
/// dedicated OS thread + current-thread executor. Tenant state stays on the task: credentials,
/// transcript, `/session` memory, persistence, tools, approvals, and exec endpoints are built inside
/// `serve_session`, not shared. The process runtime itself is `current_thread` by default (see
/// `main.rs::build_runtime`).
fn serve_session_body(base: ServeConfig) -> SessionBody {
    Box::new(move |id, service, input_rx, out_conn, running| {
        let cfg = session_cfg(&base, id, service);
        let id = id.to_owned();
        // Kept past the move into `serve_session` so a session that fails to *start* can still be
        // reported: without this the only trace of "your sandbox is unreachable" was a line on the
        // replica's stderr, and the client saw a socket that accepted its commands and answered
        // nothing.
        let out_err = out_conn.clone();
        Box::pin(async move {
            if let Err(e) = serve_session(cfg, input_rx, out_conn, running).await {
                eprintln!("serve: session {id} ended: {e}");
                lock_ignoring_poison(&out_err).broadcast(OutFrame::Value(json!({
                    "type": "error",
                    "session_id": id,
                    "error": e.to_string(),
                })));
            }
        })
    })
}

/// Take the advisory lock on a session directory, off the runtime: both the `create_dir_all` and the
/// lock itself are blocking filesystem calls, and on a network filesystem neither is fast.
///
/// The directory is created first, and that ordering is the whole point: the lock lives *inside* the
/// session directory, and [`acquire_session_lock`] falls back to a `<path>.lock` sibling for a path
/// that isn't a directory — so a replica that skipped this on a brand-new session would lock a
/// different file than the replica that found the directory already there. `create_dir_all` is
/// idempotent, and a directory with no segments is not a session, so this mints nothing.
///
/// Bounded by [`SESSION_LOCK_TIMEOUT`]: see [`take_session_lock_within`] for what happens to a lock
/// that is acquired after the deadline has already passed.
async fn take_session_lock(path: std::path::PathBuf) -> std::io::Result<Option<SessionLock>> {
    take_session_lock_within(SESSION_LOCK_TIMEOUT, move || {
        std::fs::create_dir_all(&path)?;
        acquire_session_lock(&path)
    })
    .await
}

/// [`take_session_lock`] with the blocking work and the deadline injected, so a test can drive both.
///
/// **The orphan is dropped, not handed on.** `spawn_blocking` is uncancellable, so a probe that
/// misses the deadline keeps running and may well acquire the lock afterwards — at which point
/// nobody is serving that session. Handing the acquired lock to the next attempt would mean keeping
/// a per-path registry of orphans, a second place that can leak and a second rule about who owns a
/// lock. Instead the result is delivered through a `oneshot`: once the waiter has given up, `send`
/// hands the value back to the blocking thread, where the [`SessionLock`] drops on the spot —
/// closing the descriptor (which is what releases a POSIX lock) and freeing the in-process
/// registration. The retry that follows the 503 then takes the lock the ordinary way.
async fn take_session_lock_within<F>(
    deadline: Duration,
    take: F,
) -> std::io::Result<Option<SessionLock>>
where
    F: FnOnce() -> std::io::Result<Option<SessionLock>> + Send + 'static,
{
    let (tx, rx) = oneshot::channel();
    tokio::task::spawn_blocking(move || {
        // `send` returning `Err` *is* the release: the value comes back here and drops before this
        // thread returns.
        let _ = tx.send(take());
    });
    match tokio::time::timeout(deadline, rx).await {
        Ok(Ok(result)) => result,
        // The blocking task was dropped without answering — the runtime is going away.
        Ok(Err(e)) => Err(std::io::Error::other(e)),
        Err(_) => Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            format!("taking the session lock did not finish within {deadline:?}"),
        )),
    }
}

/// Release a session's lock and, if the session never got far enough to write a segment, take the
/// directory [`take_session_lock`] had to create in order to hold that lock.
///
/// This runs **only on the creator's own failure path** — this task holds the lock, and the session
/// body has already ended — which is what makes it safe. A would-be creator evaluating a delete
/// predicate could not do this: its `rmdir` races another replica's `create_dir_all` + `O_EXCL`
/// segment create, and on NFS/EFS attribute staleness would let it delete a directory whose fresh
/// segment it simply had not seen yet. Anything this misses (a lock that could not be taken at all,
/// a replica that died mid-start) is left for an out-of-band sweep; see ARCHITECTURE.md.
///
/// `remove_dir` *is* the emptiness check, and an atomic one: `000001.jsonl` is created with `O_EXCL`
/// before anything else and never deleted, so a directory holding nothing but `lock` has never been
/// a session — and if a racing replica got one written in between, `ENOTEMPTY` leaves everything
/// alone. The `lock` file is unlinked first because `remove_dir` would otherwise always fail; the
/// window that opens between that unlink and this task's own release is harmless, since the lock is
/// liveness-only (correctness is the epoch fence) and this replica is already done with the session.
async fn release_session_lock(lock: Option<SessionLock>, path: Option<std::path::PathBuf>) {
    let (Some(lock), Some(path)) = (lock, path) else {
        return;
    };
    // Off the runtime thread: three network-filesystem calls and a descriptor close.
    let _ = tokio::task::spawn_blocking(move || {
        let only_lock = std::fs::read_dir(&path).is_ok_and(|entries| {
            entries
                .flatten()
                .all(|e| e.file_name() == std::ffi::OsStr::new("lock"))
        });
        if only_lock {
            let _ = std::fs::remove_file(path.join("lock"));
            let _ = std::fs::remove_dir(&path);
        }
        drop(lock);
    })
    .await;
}

/// Report a session that never started: to the pin that spawned it (which turns it into an HTTP
/// status), to any connection that attached in the meantime, and to the replica's own log.
fn report_start_failure(
    started: oneshot::Sender<Result<(), PinError>>,
    out_conn: &SharedOutConn,
    id: &str,
    error: PinError,
    why: &str,
) {
    eprintln!("serve: session {id} {why}");
    let message = match error {
        PinError::Forbidden => "that session belongs to another tenant",
        PinError::Unavailable(m) => m,
        PinError::Closed => "the daemon is shutting down",
    };
    // The spawning connection is answered by `started`; this reaches anyone who attached to the
    // `Starting` slot while the lock was being taken.
    lock_ignoring_poison(out_conn).broadcast(OutFrame::Value(json!({
        "type": "error",
        "session_id": id,
        "error": message,
    })));
    let _ = started.send(Err(error));
}

/// A session task's hold on its own table entry — and the only thing that ever removes one. However
/// the task ends (its body returned, panicked, or the runtime dropped it), this removes the entry if
/// it's still this incarnation's, and only **then** fires `exited`, so whoever that wakes finds the id
/// free.
struct ExitGuard {
    table: Arc<Mutex<Table>>,
    id: String,
    incarnation: u64,
    exited: CancellationToken,
    /// Counted down here rather than at any of the places a session can end, because this guard is
    /// the one thing that runs on every path out — including a panic.
    metrics: Option<Arc<crate::metrics::Metrics>>,
    /// Why this session ended, set by whoever ended it; `Client` unless something else claims it.
    end_reason: Arc<Mutex<crate::metrics::SessionEnd>>,
}

impl ExitGuard {
    /// `Starting → Live`. `false` if the entry was stopped first (a shutdown or reap raced the start),
    /// in which case the body never runs: a session stopped before it went live never touches storage.
    ///
    /// Any start-up work that has to finish before the session touches storage belongs *before* this
    /// call, awaited on the task — never under the map lock.
    fn go_live(&self) -> bool {
        let mut table = lock_ignoring_poison(&self.table);
        let Some(h) = table
            .sessions
            .get_mut(&self.id)
            .filter(|h| h.incarnation == self.incarnation)
        else {
            return false;
        };
        match std::mem::replace(&mut h.phase, Phase::Stopping) {
            Phase::Starting(tx) => {
                h.phase = Phase::Live(tx);
                true
            }
            other => {
                h.phase = other;
                false
            }
        }
    }
}

impl Drop for ExitGuard {
    fn drop(&mut self) {
        if let Some(m) = &self.metrics {
            m.sessions_live.dec();
            m.session_ended(*lock_ignoring_poison(&self.end_reason));
        }
        {
            let mut table = lock_ignoring_poison(&self.table);
            if table
                .sessions
                .get(&self.id)
                .is_some_and(|h| h.incarnation == self.incarnation)
            {
                table.sessions.remove(&self.id);
            }
        }
        self.exited.cancel();
    }
}

/// One attachment to a session, as [`Supervisor::pin`] hands it out. Its holder **must**
/// [`unpin`](Supervisor::unpin) with this `incarnation` when the attachment ends.
struct Pinned {
    id: String,
    incarnation: u64,
    input_tx: mpsc::Sender<String>,
    out_conn: SharedOutConn,
}

/// A freshly spawned session task's report to the pin that spawned it, sent before it goes live: it
/// owns the session's storage, or another owner holds it and nothing was started.
type Started = oneshot::Receiver<Result<(), PinError>>;

/// What one look at the table found for an id ([`Supervisor::try_pin`]).
enum TryPin {
    /// Attached (spawning the session if the id was free): its incarnation, input, and output. The
    /// `Started` receiver is present exactly when this look *spawned* the session and that spawn has
    /// start-up work to finish first — the pin awaits it, so a refusal is still an HTTP status.
    Attached(u64, mpsc::Sender<String>, SharedOutConn, Option<Started>),
    /// The id's previous task is still exiting. Wait for this latch, then look again.
    Wait(CancellationToken),
    /// A live session owns this id, and it belongs to another tenant.
    Forbidden,
    /// This replica already holds `--max-live-sessions`.
    AtCapacity,
    /// The daemon is shutting down.
    Closed,
}

/// Why [`Supervisor::pin`] refused. An enum rather than a message because the three cases are three
/// different HTTP statuses, and the WebSocket path answers with one **before** the upgrade — a 403
/// after a 101 would only be readable as a close code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PinError {
    /// 403: this id is live under another tenant.
    Forbidden,
    /// 503: the id's previous task is still exiting, or is wedged on its way out. Retryable.
    Unavailable(&'static str),
    /// 503: the daemon is shutting down.
    Closed,
}

impl HttpError {
    /// Which refusal bucket this answer belongs in, or `None` when it is not a refusal at all (a
    /// timeout, a 405 from a probe). Derived from the status rather than the variant so a new
    /// variant lands in the right bucket by construction.
    fn refusal(&self) -> Option<crate::metrics::Refusal> {
        use crate::metrics::Refusal;
        match self.status().0 {
            401 | 403 => Some(Refusal::Auth),
            421 => Some(Refusal::Misdirected),
            503 => Some(Refusal::Unavailable),
            400 => Some(Refusal::BadRequest),
            _ => None,
        }
    }
}

impl From<PinError> for HttpError {
    fn from(e: PinError) -> Self {
        match e {
            PinError::Forbidden => HttpError::Forbidden("that session belongs to another tenant"),
            PinError::Unavailable(why) => HttpError::Unavailable(why),
            PinError::Closed => HttpError::Unavailable("the daemon is shutting down"),
        }
    }
}

impl Supervisor {
    /// Count a refusal, then answer it.
    ///
    /// One function so a refusal path added later cannot quietly skip the counter — the alternative
    /// is a counter beside each of seven `write_http_err` calls, which is exactly the shape that
    /// goes stale. Errors that are not refusals (a 405 from a probe, a timeout) are answered without
    /// being counted; [`HttpError::refusal`] decides which is which.
    async fn refuse<S: AsyncWrite + Unpin>(&self, stream: &mut S, err: &HttpError) {
        if let (Some(m), Some(r)) = (&self.metrics, err.refusal()) {
            m.refused(r);
        }
        let _ = write_http_err(stream, err, None).await;
    }

    /// Attach to the session named `requested_id` (minting a fresh id if `None`), spawning it if no task
    /// owns the id. No eviction: multiple attachments coexist on one session (WebSocket connections
    /// and in-flight HTTP POSTs), so a phone, a TUI, and a `curl` can watch/drive it together.
    ///
    /// If the id's previous task is still exiting (`Stopping`), this **waits** for it to be gone before
    /// spawning the next one — never two tasks, and so never two writers, on one session. Bounded by
    /// [`JOIN_GRACE`]: a task that wedged on its way out refuses the attach rather than hanging it (or
    /// being joined by a rival). `Err` carries the reason for the client.
    ///
    /// The caller **must** [`unpin`](Self::unpin) when the attachment ends — a WebSocket on socket
    /// close, an HTTP POST when its response has been written — so the idle reaper's clock starts
    /// once nobody is attached.
    async fn pin(
        &self,
        requested_id: Option<String>,
        service: Option<Arc<ServiceSession>>,
    ) -> Result<Pinned, PinError> {
        self.pin_within(requested_id, service, JOIN_GRACE).await
    }

    /// [`Self::pin`], with the wait for a stopping predecessor as a parameter so it can be tested.
    async fn pin_within(
        &self,
        requested_id: Option<String>,
        service: Option<Arc<ServiceSession>>,
        grace: Duration,
    ) -> Result<Pinned, PinError> {
        let id = requested_id.unwrap_or_else(new_id);
        let deadline = tokio::time::Instant::now() + grace;
        loop {
            match self.try_pin(&id, service.as_ref()) {
                TryPin::Attached(incarnation, input_tx, out_conn, started) => {
                    // A spawn with start-up work reports before it goes live. Waiting here is what
                    // turns "another replica holds this session" into a 503 the client can act on;
                    // the task frees the id itself on the way out, so there is nothing to unpin.
                    if let Some(started) = started {
                        match started.await {
                            Ok(Ok(())) => {}
                            Ok(Err(e)) => return Err(e),
                            // The task was dropped before it reported — the runtime is going away.
                            Err(_) => return Err(PinError::Closed),
                        }
                    }
                    return Ok(Pinned {
                        id,
                        incarnation,
                        input_tx,
                        out_conn,
                    });
                }
                TryPin::Forbidden => return Err(PinError::Forbidden),
                TryPin::AtCapacity => {
                    return Err(PinError::Unavailable(
                        "this replica is at its live-session limit; retry",
                    ));
                }
                TryPin::Closed => return Err(PinError::Closed),
                TryPin::Wait(exited) => {
                    if tokio::time::timeout_at(deadline, exited.cancelled())
                        .await
                        .is_err()
                    {
                        eprintln!(
                            "serve: session {id} did not exit within {}ms of stopping; refusing to \
                             start another alongside it",
                            grace.as_millis()
                        );
                        return Err(PinError::Unavailable(
                            "the session is still stopping; retry",
                        ));
                    }
                }
            }
        }
    }

    /// One look at the table for `id`, synchronously (the lock is a std `Mutex`, never held across an
    /// `.await`): attach to its session — spawning it, `Starting`, if the id is free — or report that
    /// its previous task is still exiting.
    fn try_pin(&self, id: &str, service: Option<&Arc<ServiceSession>>) -> TryPin {
        let tenant = service.map(|svc| svc.tenant().to_owned());
        let mut table = lock_ignoring_poison(&self.table);
        if table.lifecycle == Lifecycle::Closed {
            return TryPin::Closed;
        }
        if let Some(h) = table.sessions.get_mut(id) {
            // Checked before anything else about the slot: an id alone never grants access to a
            // running session. A `Stopping` slot is refused just the same — its successor would
            // otherwise be spawned by, and inherit the grant of, the wrong tenant.
            if h.tenant != tenant {
                return TryPin::Forbidden;
            }
            return match h.phase.input() {
                // Register, don't evict: one more attached connection, and clear any detach timestamp
                // so the reaper's clock only runs while genuinely detached (`attached == 0`).
                Some(tx) if !tx.is_closed() => {
                    let input_tx = tx.clone();
                    h.attached += 1;
                    h.last_detached_at = None;
                    // No `Started`: this task is already running (or already past its own lock), so
                    // there is nothing for the caller to wait on.
                    TryPin::Attached(h.incarnation, input_tx, h.out_conn.clone(), None)
                }
                // Stopping — or attachable in name only: a closed input means its loop already ended on
                // its own (SIGTERM, an internal error) and the task is on its way out, possibly still
                // persisting. Respawning now would put two writers on one session file; mark it
                // stopping (idempotent) and wait for it to be gone.
                _ => {
                    h.stop(crate::metrics::SessionEnd::Error);
                    TryPin::Wait(h.exited.clone())
                }
            };
        }

        // Nothing owns this id, so this look is about to spawn one. Everything from here is a
        // *new-session* gate, which is why the drain check sits here and not at the top of this
        // function: a draining replica still owns live sessions, and a reconnect to one of those
        // took the `get_mut` branch above and never reaches this line. Refusing at the top — which
        // an earlier draft did — would refuse exactly the clients the drain exists to protect.
        if table.lifecycle != Lifecycle::Open {
            return TryPin::Closed;
        }

        // Checked here, under the same lock the insert happens under, so two simultaneous
        // connections can't both see room for the last slot.
        if let Some(max) = self.service.as_ref().and_then(|s| s.max_live_sessions)
            && table.sessions.len() >= max
        {
            return TryPin::AtCapacity;
        }

        let incarnation = table.next_incarnation;
        table.next_incarnation += 1;
        let (input_tx, input_rx) = mpsc::channel::<String>(crate::serve::IN_CHANNEL_BOUND);
        // Made before the entry so the handle and the exit guard share one slot: whoever ends the
        // session writes the reason here, and the guard reports it on the way out.
        let end_reason = Arc::new(Mutex::new(crate::metrics::SessionEnd::Client));
        let out_conn: SharedOutConn = Arc::new(Mutex::new(OutFanout::default()));
        // Shared with the session loop: `true` only while it's running a `prompt`. The reaper reads
        // this handle-side clone to never reclaim a mid-run background session.
        let running = Arc::new(AtomicBool::new(false));
        let exited = CancellationToken::new();
        table.sessions.insert(
            id.to_owned(),
            SessionHandle {
                incarnation,
                phase: Phase::Starting(input_tx.clone()),
                tenant,
                out_conn: out_conn.clone(),
                exited: exited.clone(),
                attached: 1,
                last_detached_at: None,
                running: running.clone(),
                end_reason: Arc::clone(&end_reason),
            },
        );
        // The entry now owns the id; everything below runs unlocked. That includes the spawn itself: a
        // task tokio drops on the spot (a runtime shutting down) drops its `ExitGuard`, which takes
        // this lock.
        drop(table);

        // Counted at the moment the id is owned, not when the body starts: from here on this
        // session is one of the things a crash of this replica would strand until its lock lapses,
        // which is what the gauge is for.
        if let Some(m) = &self.metrics {
            m.sessions_live.inc();
            m.sessions_spawned.inc();
        }
        let exit = ExitGuard {
            table: Arc::clone(&self.table),
            id: id.to_owned(),
            incarnation,
            exited,
            metrics: self.metrics.clone(),
            end_reason: Arc::clone(&end_reason),
        };
        // Service mode: this task must own the session's storage before it goes live. The lock is
        // liveness only — correctness is the epoch fence — so it is what keeps two replicas sharing a
        // mount from both replaying and then fencing each other turn after turn.
        let lock_path = service.map(|svc| svc.session_path());
        let (started_tx, started_rx) = oneshot::channel();
        let started = lock_path.is_some().then_some(started_rx);
        let body = (self.body)(id, service.cloned(), input_rx, out_conn.clone(), running);
        let session_id = id.to_owned();
        // For the failure paths below: whoever attached to the `Starting` slot while the lock was
        // being taken is told why nothing started.
        let starting_conn = out_conn.clone();
        let metrics_for_lock = self.metrics.clone();
        tokio::spawn(async move {
            let mut lock = None;
            if let Some(path) = lock_path.clone() {
                // Timed because this is where a dead owner's lease shows up. A `kill -9` does not
                // release an NFS lock — it lapses — so after a replica dies its sessions wait here
                // rather than failing, and the wait is the whole failover story. `lock_failures`
                // separates "another live replica holds it" (ordinary, a 503 and a retry) from a
                // mount that stopped answering.
                let began = std::time::Instant::now();
                let outcome = take_session_lock(path).await;
                if let Some(m) = &metrics_for_lock {
                    match &outcome {
                        Ok(Some(_)) => m.lock_wait_seconds.observe(began.elapsed().as_secs_f64()),
                        Ok(None) => m.lock_failed(crate::metrics::LockFailure::Held),
                        Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
                            m.lock_failed(crate::metrics::LockFailure::Timeout);
                        }
                        Err(_) => m.lock_failed(crate::metrics::LockFailure::Io),
                    }
                }
                match outcome {
                    Ok(Some(held)) => lock = Some(held),
                    Ok(None) => {
                        report_start_failure(
                            started_tx,
                            &starting_conn,
                            &session_id,
                            PinError::Unavailable("that session is open on another replica; retry"),
                            "is held elsewhere",
                        );
                        drop(body);
                        drop(exit);
                        return;
                    }
                    Err(e) => {
                        report_start_failure(
                            started_tx,
                            &starting_conn,
                            &session_id,
                            PinError::Unavailable("that session's storage is unavailable; retry"),
                            &format!("could not be locked: {e}"),
                        );
                        drop(body);
                        drop(exit);
                        return;
                    }
                }
            }
            let _ = started_tx.send(Ok(()));
            if exit.go_live() {
                body.await;
            } else {
                drop(body);
            }
            // Before the `ExitGuard`: that guard frees the id and wakes whoever is waiting for it, and
            // the next owner's first move is to take this very lock. A session that never wrote a
            // segment also gives its directory back here — the one path where that is safe.
            release_session_lock(lock, lock_path).await;
            // Only once the body's state is gone: free the id, then wake whoever waits on it.
            drop(exit);
        });
        TryPin::Attached(incarnation, input_tx, out_conn, started)
    }

    /// One fewer attached connection; if that was the last, start the idle reaper's clock. A no-op for
    /// any incarnation but the one pinned — a connection outliving its session must not detach the
    /// next one under the same id.
    fn unpin(&self, id: &str, incarnation: u64) {
        let mut table = lock_ignoring_poison(&self.table);
        if let Some(h) = table.sessions.get_mut(id)
            && h.incarnation == incarnation
        {
            h.attached = h.attached.saturating_sub(1);
            if h.attached == 0 {
                h.last_detached_at = Some(Instant::now());
            }
        }
    }

    /// Drive `ws` against an **already pinned** session until the socket closes — the session itself
    /// keeps running either way.
    ///
    /// The pin happens before the upgrade (see [`handle_websocket_upgrade`]), not here: a refusal has
    /// to be a real HTTP status. A 403 or 503 delivered as a close code on an accepted WebSocket is a
    /// successful handshake followed by a hang-up, which no HTTP client, proxy, or retry policy can
    /// read as "wrong tenant" or "try again".
    async fn attach<S>(
        self: &Arc<Self>,
        pinned: Pinned,
        service: Option<Arc<ServiceSession>>,
        ws: WebSocketStream<S>,
    ) where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let Pinned {
            id,
            incarnation,
            input_tx,
            out_conn,
        } = pinned;

        // Register this connection's send channel as one of the session's output sinks — the session
        // broadcasts every frame to all registered sinks. Keep the `sink_id` to remove it on disconnect.
        let (conn_tx, mut conn_rx) = mpsc::channel::<OutFrame>(OUT_CHANNEL_BOUND);
        // A direct handle to *this* connection's send channel, for supervisor-level replies (the
        // `list_daemon_sessions` command below) that must go back to *this* socket only, not fan out to
        // the other attached connections. Feeds the same `conn_rx`/send task.
        let reply_tx = conn_tx.clone();
        // `add_with_catchup`, not `add`: the session's committed history is queued on this connection
        // *ahead of* its sink going live, so a client re-attaching to a run already in flight receives
        // what it missed before that run's live frames — not after them, and not never. Frames carry no
        // sequence number, so delivery order is the only order a client has; leaving catch-up to a
        // client round trip issued after connecting raced the very stream it was meant to precede. Both
        // halves happen under the fanout's lock (which `broadcast` also takes), so the ordering is
        // structural rather than lucky — see `OutFanout::add_with_catchup`.
        let sink_id = lock_ignoring_poison(&out_conn).add_with_catchup(OutSink::Bounded(conn_tx));

        let (mut sink, mut stream) = ws.split();

        // The send task owns the socket's write half. Everything outbound — protocol frames, `Pong`
        // replies, and periodic keepalive `Ping`s — funnels through it so the single sink has one
        // writer. `ctrl_rx` carries the read loop's `Pong` replies, bounded like every other queue a
        // client can push into (see [`CTRL_CHANNEL_BOUND`]).
        let (ctrl_tx, mut ctrl_rx) = mpsc::channel::<Message>(CTRL_CHANNEL_BOUND);
        // Stops this connection's send task at teardown (when its read loop ends on socket close). The
        // send task also stops on its own if `conn_rx` closes (this connection's sink removed + `reply_tx`
        // dropped).
        let send_cancel = CancellationToken::new();
        let send_task_cancel = send_cancel.clone();
        let send_task = tokio::spawn(async move {
            let mut ping = tokio::time::interval(PING_INTERVAL);
            ping.set_missed_tick_behavior(MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    biased;
                    _ = send_task_cancel.cancelled() => break,
                    // A closed `conn_rx` means we were superseded (out_conn rebound) or the session
                    // ended — stop writing to this socket.
                    // A closed `conn_rx` means this connection's sink was removed (teardown) — stop.
                    frame = conn_rx.recv() => match frame {
                        Some(frame) => if let Some(line) = frame_to_line(frame) {
                            // `line` is already valid UTF-8 (built from a `String`); `try_from` only
                            // re-validates — no copy — to hand the same `Bytes` to a text frame.
                            match Utf8Bytes::try_from(line) {
                                Ok(text) => if sink.send(Message::Text(text)).await.is_err() {
                                    break;
                                },
                                Err(e) => eprintln!("serve: non-utf8 output frame skipped: {e}"),
                            }
                        },
                        None => break,
                    },
                    ctrl = ctrl_rx.recv() => {
                        if let Some(msg) = ctrl
                            && sink.send(msg).await.is_err() {
                                break;
                            }
                    },
                    _ = ping.tick() => {
                        if sink.send(Message::Ping(Vec::new().into())).await.is_err() {
                            break;
                        }
                    }
                }
            }
            let _ = sink.close().await;
        });

        // Read loop: this socket's inbound messages. Each text message is exactly one command line fed
        // into the session; the session (not this loop) decides what to do with it. Ends when the socket
        // closes — there's no eviction, so a peer connection never tears this one down.
        loop {
            tokio::select! {
                biased;
                msg = stream.next() => {
                    let Some(msg) = msg else { break }; // socket closed
                    let msg = match msg {
                        Ok(m) => m,
                        Err(_) => break,
                    };
                    match msg {
                        Message::Text(text) => {
                            // Supervisor-level intercept: `list_daemon_sessions` is answered *here* (only
                            // the supervisor sees every session), not forwarded to the pinned session. A
                            // cheap substring prefilter keeps the byte-identical hot path allocation-free
                            // for every other command; only a real match pays the parse.
                            if text.contains("list_daemon_sessions")
                                && let Ok(v) = serde_json::from_str::<serde_json::Value>(text.as_str())
                                    && v.get("type").and_then(serde_json::Value::as_str)
                                        == Some("list_daemon_sessions")
                                    {
                                        let client_id = v
                                            .get("id")
                                            .and_then(serde_json::Value::as_str)
                                            .map(str::to_owned);
                                        let frame = self
                                            .list_daemon_sessions(client_id, service.as_deref())
                                            .await;
                                        let _ = reply_tx.try_send(frame);
                                        continue;
                                    }
                            // Never dropped, never coalesced: a client that got no error believes its
                            // command was accepted, so discarding one leaves it waiting forever on a
                            // response that will never come — a dropped command is a correctness bug,
                            // not a shed load. So this *awaits* a full queue rather than shedding: the
                            // read loop stops pulling from the socket, TCP's own window closes, and the
                            // backpressure lands where it belongs — on the client that is outrunning the
                            // session loop. Any one command is separately bounded by
                            // [`MAX_INBOUND_MESSAGE_BYTES`], so a full queue is bounded in bytes too.
                            if input_tx.send(text.as_str().to_owned()).await.is_err() {
                                break; // session gone
                            }
                        }
                        Message::Ping(payload) => {
                            // Drop-on-full (see [`CTRL_CHANNEL_BOUND`]): a `Pong` is a courtesy, and a
                            // `Ping` flood from a client that never drains its read side must not be
                            // able to queue work faster than the send task can write it out.
                            let _ = ctrl_tx.try_send(Message::Pong(payload));
                        }
                        Message::Close(_) => break,
                        // Pong (keepalive ack), Binary (protocol is text JSON), and any future frame
                        // kind are ignored.
                        _ => {}
                    }
                }
            }
        }

        // This connection is done. Remove its sink from the fanout (the session stops writing to this
        // socket) and stop its send task. The session keeps running — other connections stay attached,
        // and even the last one leaving just detaches (the run lives on for a reconnect).
        lock_ignoring_poison(&out_conn).remove(sink_id);
        send_cancel.cancel();
        let _ = send_task.await;

        // One fewer attached connection; if that was the last, start the idle reaper's clock.
        self.unpin(&id, incarnation);
    }

    /// Drain: stop taking new work, let what is in flight finish, then shut down.
    ///
    /// The sequence, and why each step is where it is:
    ///
    /// 1. **`Draining` immediately.** `/readyz` starts answering 503 on the next probe, so the load
    ///    balancer stops choosing this replica within one probe interval, while `/livez` stays 200 so
    ///    the orchestrator doesn't decide the process is wedged and kill it mid-drain. New sessions
    ///    get a 503 and the edge's retry finds another replica; **reconnects to sessions this replica
    ///    still owns keep working**, because those clients cannot be served anywhere else until this
    ///    replica releases the session's lock.
    /// 2. **Wait for in-flight runs**, bounded by `grace`. The wait is on each session's `running`
    ///    flag — true exactly while a prompt is executing. It is deliberately **not** `is_reapable`,
    ///    which additionally requires `attached == 0`: a client watching its own run keeps a socket
    ///    attached throughout, so a drain gated on that would burn its entire grace every time and
    ///    then kill the run anyway.
    /// 3. **Then the ordinary shutdown**, which stops every session and waits for the tasks.
    ///
    /// `grace` of zero skips step 2 entirely, which is the behaviour a single-user daemon had before
    /// this existed: Ctrl-C ends it now, not in thirty seconds.
    fn begin_drain(&self) {
        lock_ignoring_poison(&self.table).lifecycle = Lifecycle::Draining;
    }

    /// Is any session still executing a prompt?
    ///
    /// Reads each session's `running` flag — true exactly while a prompt is executing. Deliberately
    /// **not** `is_reapable`, which additionally requires `attached == 0`: a client watching its own
    /// run holds a socket open throughout, so a drain gated on that would burn its whole grace every
    /// time and then cut the run anyway.
    fn any_run_in_flight(&self) -> bool {
        lock_ignoring_poison(&self.table)
            .sessions
            .values()
            .any(|h| h.running.load(Ordering::Relaxed))
    }

    /// Graceful shutdown: close the table to new attachments, stop every session (dropping each
    /// retained input, which closes that session's input channel → it cancels any in-flight run,
    /// persists, and exits), then wait for every session task to be gone so persistence is durable
    /// before the process exits — including any the reaper already stopped that are still on their way
    /// out. Bounded so a wedged session can't hang the shutdown forever — a straggler is left to
    /// `process::exit`.
    async fn shutdown(&self) {
        let exits: Vec<CancellationToken> = {
            let mut table = lock_ignoring_poison(&self.table);
            table.lifecycle = Lifecycle::Closed;
            table
                .sessions
                .values_mut()
                .map(|h| {
                    h.stop(crate::metrics::SessionEnd::Drain);
                    h.exited.clone()
                })
                .collect()
        };
        await_exits_within(exits, JOIN_GRACE).await;
    }

    /// Answer a `list_daemon_sessions` command: the union of every session the daemon knows about — the
    /// live in-memory map (`live:true`) and every `*.jsonl` under the base `--session-dir` (whose
    /// `live` flag says whether that persisted id also has a running task right now). Persisted
    /// entries come back newest first (`scan_session_dirs`, the same order `list_sessions` and
    /// `list_all_sessions` use); a live session with nothing on disk yet is appended after them. The
    /// reply is a single `response` frame the caller sends back on the originating connection.
    /// **Tenant-scoped in service mode**, on both halves: only this tenant's live handles, and only
    /// `<every mounted shard>/<tenant>/sessions` on disk. The supervisor sees every session on the
    /// replica, so without the scope this one command would enumerate the whole fleet-mate set.
    async fn list_daemon_sessions(
        &self,
        client_id: Option<String>,
        service: Option<&ServiceSession>,
    ) -> OutFrame {
        // Snapshot the live ids under the lock, then drop it — the on-disk scan below must not run while
        // holding the map mutex. Live means attachable with a loop still reading its input: a
        // `Stopping` entry is on its way out, so it reports `live:false` although its task still exists.
        let tenant = service.map(|svc| svc.tenant());
        let live: HashSet<String> = {
            let table = lock_ignoring_poison(&self.table);
            table
                .sessions
                .iter()
                .filter(|(_, h)| h.tenant.as_deref() == tenant)
                .filter(|(_, h)| h.phase.input().is_some_and(|tx| !tx.is_closed()))
                .map(|(id, _)| id.clone())
                .collect()
        };

        // On-disk listings, if this daemon persists at all. Through `scan_session_dirs`, so the
        // `read_dir` goes to the blocking pool along with the listing parse — this is the supervisor,
        // answering on a connection task that shares the one runtime thread with every live session,
        // so a shard that is slow to enumerate must not be enumerated here.
        let dirs: Vec<std::path::PathBuf> = match (service, &self.session_dir) {
            (Some(svc), _) => svc.tenant_session_dirs(),
            (None, Some(dir)) => vec![std::path::PathBuf::from(dir)],
            (None, None) => Vec::new(),
        };
        // A tenant's sessions are epoch segments sealed with its own key; the daemon's own
        // `--session-dir` is the single-file layout it always was.
        let layout = service.map_or(Layout::File, ServiceSession::layout);
        let metas = if dirs.is_empty() {
            Vec::new()
        } else {
            crate::serve::scan_session_dirs(dirs, layout, |_, _| {}).await
        };

        let mut seen: HashSet<&str> = HashSet::with_capacity(metas.len());
        let mut sessions: Vec<serde_json::Value> = Vec::with_capacity(metas.len().max(live.len()));
        for meta in &metas {
            let mut obj = meta.to_listing_json();
            if let serde_json::Value::Object(map) = &mut obj {
                map.insert("live".into(), json!(live.contains(&meta.id)));
            }
            seen.insert(meta.id.as_str());
            sessions.push(obj);
        }
        // A brand-new live session may have no file yet (in-memory only, or not yet flushed) — surface
        // it too, as a minimal entry, so a client sees every reachable session.
        for id in &live {
            if !seen.contains(id.as_str()) {
                sessions.push(json!({ "id": id, "live": true }));
            }
        }

        OutFrame::Value(json!({
            "type": "response",
            "command": "list_daemon_sessions",
            "id": client_id,
            "success": true,
            "data": { "sessions": sessions },
        }))
    }

    /// Idle reaper: stop every session [`is_reapable`] names. Stopping drops its retained input, so the
    /// session observes EOF and persists+exits exactly as in [`Self::shutdown`]; its entry stays
    /// (`Stopping`) until the task is gone, which is what makes a reconnect in the meantime wait for it
    /// rather than start a rival. Nothing here waits: the task frees its own id on exit, and a
    /// reconnect to it then respawns and replays from disk.
    fn reap_idle(&self, timeout: Duration) {
        let mut table = lock_ignoring_poison(&self.table);
        for h in table.sessions.values_mut() {
            if is_reapable(h, timeout) {
                h.stop(crate::metrics::SessionEnd::IdleReap);
            }
        }
    }
}

/// Whether the reaper should stop this session.
///
/// A `Stopping` session is already on its way out: nothing to do.
///
/// A **closed input** is the strongest reason of all: the session's loop is already gone (it returned
/// early — no credential, an unwritable session dir — or hit an internal error), so nothing will ever
/// read its input again. Stop it whatever its attach state or clock says — a live connection still
/// pinned to it is no reason to keep a corpse attachable (that connection's next command fails its
/// send and tears the socket down; a reconnect waits for the task to finish and respawns the id).
///
/// Otherwise the session is alive, and the ordinary conditions apply: **detached** (`attached == 0`) for
/// at least `timeout`, and not mid-`prompt` — a detached background run is exactly what this design
/// exists to keep alive, so `running` is never reaped out from under an in-flight turn.
fn is_reapable(h: &SessionHandle, timeout: Duration) -> bool {
    match h.phase.input() {
        None => false,
        Some(tx) if tx.is_closed() => true,
        Some(_) => {
            h.attached == 0
                && h.last_detached_at.is_some_and(|d| d.elapsed() >= timeout)
                && !h.running.load(Ordering::Relaxed)
        }
    }
}

/// Wait for a batch of session tasks to exit — each one's [`SessionHandle::exited`] latch — bounded by
/// `grace` so a wedged session can't hold the caller forever. A straggler past `grace` is left running
/// (and still frees its id whenever it does exit); all a graceful shutdown can do then is fall through
/// to `process::exit`.
async fn await_exits_within(exits: Vec<CancellationToken>, grace: Duration) {
    let all = futures::future::join_all(exits.iter().map(CancellationToken::cancelled));
    if tokio::time::timeout(grace, all).await.is_err() {
        let stragglers = exits.iter().filter(|e| !e.is_cancelled()).count();
        eprintln!("serve: {stragglers} session(s) did not persist within the join grace period");
    }
}

/// Resolve the idle reaper's window from [`ServeConfig::session_idle_timeout`]: unset ⇒ the default
/// for this mode ([`DEFAULT_SESSION_IDLE_TIMEOUT`], or
/// [`DEFAULT_SERVICE_SESSION_IDLE_TIMEOUT`] on a replica — see that const for why an hour is the
/// wrong number there), and `0` ⇒ `None`, the explicit opt-out for an operator who genuinely wants
/// every session pinned for the daemon's lifetime.
///
/// An operator who passes a value still gets exactly it, in either mode. The mode only chooses what
/// *silence* means.
fn resolve_idle_timeout(configured: Option<Duration>, service_mode: bool) -> Option<Duration> {
    match configured {
        Some(t) if t.is_zero() => None,
        Some(t) => Some(t),
        None if service_mode => Some(DEFAULT_SERVICE_SESSION_IDLE_TIMEOUT),
        None => Some(DEFAULT_SESSION_IDLE_TIMEOUT),
    }
}

/// Which transports [`serve_ws`] should bind, on **one** shared supervisor. Loopback TCP
/// ([`Self::tcp`]) and a Unix-domain socket ([`Self::uds`]) can both be on: a session created over
/// either is reachable over the other by the same `?session_id=` (they share the session map). Shaped
/// so a future systemd socket-activation arm (`systemd_fd`) is a one-field add.
pub struct ServeListeners {
    /// Loopback/internal TCP address to bind, if any. The agent authenticates no caller over TCP.
    pub tcp: Option<SocketAddr>,
    /// Unix-domain socket path to bind, if any — kernel-enforced local authz via filesystem perms.
    #[cfg(unix)]
    pub uds: Option<std::path::PathBuf>,
    /// Octal mode to `chmod` the UDS to after binding (default `0o600`). Ignored when `uds` is `None`.
    #[cfg(unix)]
    pub uds_mode: Option<u32>,
    /// Adopt a listener socket **systemd passed us** via socket activation (`LISTEN_FDS`) instead of
    /// binding our own. systemd owns the socket's lifecycle and perms, and it outlives a
    /// `systemctl restart` (connections queue in the kernel — zero-downtime). Set by `main.rs` when it
    /// detects the activation env. Composes with [`Self::tcp`]/[`Self::uds`]: an inherited fd is used
    /// for its transport; any transport without an inherited fd still binds normally.
    pub systemd: bool,
}

/// Await the next TCP connection, or `pending()` forever when there's no TCP listener — so this can
/// sit in a `select!` arm unconditionally without `select!` needing to branch on the `Option`.
async fn accept_tcp(listener: &Option<TcpListener>) -> Option<TcpStream> {
    match listener {
        Some(l) => match l.accept().await {
            Ok((stream, _peer)) => Some(stream),
            Err(e) => {
                eprintln!("serve: websocket accept failed: {e}");
                None
            }
        },
        None => std::future::pending().await,
    }
}

/// Bind a `UnixListener` at `path`, without clobbering a **live** daemon: on `AddrInUse`, probe by
/// connecting — a successful connect means another daemon owns the socket (hard error, don't remove);
/// a refused/absent connect means the socket is stale, so remove it and rebind once. After a
/// successful bind, `chmod` the socket to `mode` (default `0o600`).
#[cfg(unix)]
async fn bind_uds(
    path: &std::path::Path,
    mode: Option<u32>,
) -> Result<UnixListener, Box<dyn std::error::Error>> {
    use std::io::ErrorKind;
    use std::os::unix::fs::PermissionsExt;

    let listener = match UnixListener::bind(path) {
        Ok(l) => l,
        Err(e) if e.kind() == ErrorKind::AddrInUse => {
            // Probe: does something answer at this path?
            if UnixStream::connect(path).await.is_ok() {
                return Err(format!(
                    "unix socket {} is already in use by a live daemon",
                    path.display()
                )
                .into());
            }
            // Stale socket (connect refused / not found) — safe to remove and rebind once.
            std::fs::remove_file(path)?;
            UnixListener::bind(path)?
        }
        Err(e) => return Err(e.into()),
    };
    std::fs::set_permissions(path, PermissionsExt::from_mode(mode.unwrap_or(0o600)))?;
    Ok(listener)
}

/// Adopt the listener socket systemd passed via socket activation (`LISTEN_FDS`/`LISTEN_PID`). Returns
/// `(tcp, uds)` — whichever transport systemd activated. `listenfd::ListenFd::from_env` reads the env
/// honoring the `LISTEN_PID == getpid()` guard (so we never grab a parent's fds), and hands back the
/// inherited std listeners; we set them non-blocking and wrap as tokio listeners. Tries a unix socket
/// first (the local-daemon case), then TCP. Errors if systemd set `LISTEN_FDS` but passed nothing we
/// can serve. Must run on a tokio runtime — `from_std` registers the listener with the reactor.
#[cfg(unix)]
fn adopt_systemd_listeners()
-> Result<(Option<TcpListener>, Option<UnixListener>), Box<dyn std::error::Error>> {
    let mut fds = listenfd::ListenFd::from_env();
    // `.ok().flatten()` so a wrong-type fd (e.g. asking for unix when it's tcp) falls through rather
    // than propagating — we then try the other type.
    if let Some(std_uds) = fds.take_unix_listener(0).ok().flatten() {
        std_uds.set_nonblocking(true)?;
        eprintln!("serve: adopted systemd-activated unix socket (path {WS_PATH})");
        return Ok((None, Some(UnixListener::from_std(std_uds)?)));
    }
    if let Some(std_tcp) = fds.take_tcp_listener(0).ok().flatten() {
        std_tcp.set_nonblocking(true)?;
        let local = std_tcp.local_addr()?;
        eprintln!("serve: adopted systemd-activated tcp socket {local} (path {WS_PATH})");
        return Ok((Some(TcpListener::from_std(std_tcp)?), None));
    }
    Err(
        "systemd socket activation was requested (LISTEN_FDS set) but systemd passed no usable \
         socket file descriptor"
            .into(),
    )
}

/// Build the one `reqwest::Client` every session in this daemon shares (W3), from
/// [`ServeConfig::upstream_http2`]. `Off` returns `None` (each session keeps building its own client,
/// today's behavior). `H2c`/`Auto` return `Some(shared)` — the difference is only
/// `http2_prior_knowledge`:
///
/// - `H2c` pins the client to HTTP/2 cleartext, so all sessions multiplex over ~one TCP connection.
///   **Requires a gateway that speaks h2c** — against an h1-only gateway *every* request fails, which
///   is the footgun `serve_ws` logs loudly and why `Off` is the default.
/// - `Auto` omits prior-knowledge: HTTP/1.1 pooling today, transparently negotiating h2 if the hop
///   later moves to `https://` with ALPN — safe against any gateway.
///
/// The read (idle) timeout mirrors [`GatewayClient::with_idle_timeout`]'s: `--idle-timeout-ms` if set,
/// else the gateway's own 600s upstream idle. Fixed here at construction because the shared client is
/// injected past the `http_shared` guard, so `with_idle_timeout` will *not* rebuild it downstream.
fn build_shared_h2_client(cfg: &ServeConfig) -> Result<Option<reqwest::Client>, String> {
    // Matches `agent_core::client`'s own `READ_TIMEOUT` (private there) — a downstream hop's idle
    // patience must be at least its upstream's, which the gateway sets to 600s.
    const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(600);
    let read_timeout = cfg
        .idle_timeout_ms
        .map(Duration::from_millis)
        .unwrap_or(DEFAULT_READ_TIMEOUT);
    let build = |prior_knowledge: bool| -> Result<reqwest::Client, String> {
        agent_core::ensure_provider();
        let mut builder = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(read_timeout)
            .pool_idle_timeout(Duration::from_secs(90));
        if prior_knowledge {
            builder = builder.http2_prior_knowledge();
        }
        builder.build().map_err(|e| e.to_string())
    };
    match cfg.upstream_http2 {
        UpstreamHttp2::Off => Ok(None),
        UpstreamHttp2::H2c => Ok(Some(build(true)?)),
        UpstreamHttp2::Auto => Ok(Some(build(false)?)),
    }
}

/// Serve the control protocol over the WebSocket listeners in `listeners` (loopback TCP and/or a
/// Unix-domain socket), all fronting **one** shared supervisor so a session is reachable over either
/// transport by its id. Runs until an OS shutdown signal, returning it so the caller exits with the
/// matching code — same contract as [`serve`](crate::serve::serve). Each accepted connection is
/// handled concurrently and routed to its session by id.
pub async fn serve_ws(
    mut cfg: ServeConfig,
    listeners: ServeListeners,
) -> Result<Option<Signal>, Box<dyn std::error::Error>> {
    // W3 (shared upstream pool): build the daemon-wide client once, here, before any session spawns —
    // `session_cfg` then clones it (a cheap `Arc` bump) into every session. Log the chosen mode loudly
    // on stderr (the protocol never uses stderr): `h2c` against an h1-only gateway fails *every*
    // request, so an operator who flipped this on must see the mode they're running.
    cfg.shared_http = build_shared_h2_client(&cfg)?;
    match cfg.upstream_http2 {
        UpstreamHttp2::Off => {
            eprintln!("serve: upstream-http2=off — each session opens its own connection pool")
        }
        UpstreamHttp2::Auto => eprintln!(
            "serve: upstream-http2=auto — one shared client, HTTP/1.1 pooling (h2 if the hop gains ALPN)"
        ),
        UpstreamHttp2::H2c => eprintln!(
            "serve: upstream-http2=h2c — one shared HTTP/2-cleartext client; REQUIRES an h2c-capable gateway, else every request fails"
        ),
    }

    // systemd socket activation (a Linux/systemd feature, so unix-only): adopt the already-bound
    // listener systemd handed us via `LISTEN_FDS` rather than binding our own. Mutually exclusive with
    // `--listen`/`--listen-uds` (main.rs only sets `systemd` when neither is given). systemd owns the
    // socket, so it survives a `systemctl restart` (connections queue in the kernel) and we never
    // bind/chmod/unlink it.
    #[cfg(unix)]
    let (adopted_tcp, adopted_uds) = if listeners.systemd {
        adopt_systemd_listeners()?
    } else {
        (None, None)
    };
    #[cfg(not(unix))]
    let adopted_tcp: Option<TcpListener> = None;

    let tcp_listener = match adopted_tcp {
        Some(listener) => Some(listener),
        None => match listeners.tcp {
            Some(addr) => {
                let listener = TcpListener::bind(addr).await?;
                let local = listener.local_addr()?;
                // A well-defined line (on stderr, which the protocol never uses) so an operator — or a
                // test that binds port 0 — can learn the actual bound address.
                eprintln!("serve: websocket listening on {local} (path {WS_PATH})");
                Some(listener)
            }
            None => None,
        },
    };

    // Only a socket WE bound is ours to unlink on shutdown — never a systemd-owned one.
    #[cfg(unix)]
    let uds_path = if listeners.systemd {
        None
    } else {
        listeners.uds.clone()
    };
    #[cfg(unix)]
    let uds_listener = match adopted_uds {
        Some(listener) => Some(listener),
        None => match &listeners.uds {
            Some(path) => {
                let listener = bind_uds(path, listeners.uds_mode).await?;
                eprintln!(
                    "serve: unix socket listening on {} (path {WS_PATH})",
                    path.display()
                );
                Some(listener)
            }
            None => None,
        },
    };
    // On a non-unix target there is no UDS to bind (`--listen-uds` errored before we got here); the
    // `select!` UDS arm still needs a listener binding, so give it a `None` that `pending()`s forever.
    #[cfg(not(unix))]
    let uds_listener: Option<()> = None;

    // Read before `cfg` is moved into the supervisor: how aggressively to reap idle sessions (and
    // whether the operator opted out of reaping altogether).
    let idle_timeout = resolve_idle_timeout(cfg.session_idle_timeout, cfg.service_mode);

    // Service mode: every connection is verified against this keyring, and a tenant's sessions live
    // on these mounts. `main.rs` has already refused `--service` without both, so a `None` verifier
    // here would be a wiring bug — and one that fails **open**, so it refuses to serve instead.
    let service = if cfg.service_mode {
        let verifier = cfg.grant_verifier.clone().ok_or(
            "--service needs a session-grant verifier (--grant-key and --seal-key)".to_string(),
        )?;
        eprintln!(
            "serve: service mode — every connection must present a verified session grant; shards: {}",
            cfg.shards
                .iter()
                .map(|(name, _)| name)
                .collect::<Vec<_>>()
                .join(", ")
        );
        Some(ServiceSupervisor {
            verifier,
            shards: cfg.shards.clone(),
            // `0` turns the cap off, matching `--session-idle-timeout`'s own convention.
            max_live_sessions: (cfg.max_live_sessions > 0).then_some(cfg.max_live_sessions),
            ready: ReadyCache::default(),
        })
    } else {
        None
    };
    // Read before `cfg` is moved into the session body factory inside this initializer.
    let cfg_drain_grace = cfg.drain_grace;
    let supervisor = Arc::new(Supervisor {
        table: Arc::default(),
        // Service mode lists per tenant, from the shards, never from one process-wide directory.
        session_dir: (!cfg.service_mode)
            .then(|| cfg.session_dir.clone())
            .flatten(),
        service,
        metrics: cfg.metrics.clone(),
        body: serve_session_body(cfg),
    });
    let mut shutdown = crate::serve::ShutdownSignal::new()?;

    // The idle reaper (on unless `--session-idle-timeout 0` turned it off). A background ticker that
    // stops dead and detached-idle-not-mid-run sessions — the same `Stopping` transition (drop the
    // retained input → the session persists and exits) shutdown uses. Its handle is aborted on
    // shutdown so the process can exit cleanly. Tick at half the timeout (so a session is reaped within
    // ~1.5× the timeout at worst), capped at 30s so a long timeout still ticks at a sane cadence.
    if idle_timeout.is_none() {
        eprintln!(
            "serve: idle-session reaper OFF (--session-idle-timeout 0) — every session, including one \
             no client ever re-attaches to, holds its task and gateway client until the daemon stops"
        );
    }
    let reaper = idle_timeout.map(|t| {
        let supervisor = supervisor.clone();
        eprintln!("serve: idle-session reaper on ({}s)", t.as_secs());
        tokio::spawn(async move {
            // Floor the period at 1ms so a pathological sub-2s timeout can't hand `interval` a zero
            // period (which panics).
            let period = (t / 2)
                .min(Duration::from_secs(30))
                .max(Duration::from_millis(1));
            let mut tick = tokio::time::interval(period);
            tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
            // The first tick fires immediately; skip it so a just-started session gets a full period.
            tick.tick().await;
            loop {
                tick.tick().await;
                supervisor.reap_idle(t);
            }
        })
    });

    // Set once a signal has arrived: which signal to report on exit, and when the grace runs out.
    // Holding this instead of returning from the signal arm is the whole of the drain — **the accept
    // loop has to keep running while draining**, or the replica can answer neither the `/readyz`
    // probe that tells the load balancer to stop choosing it, nor a reconnect from a client whose
    // session this replica still owns. Draining inline and returning, which the obvious
    // implementation does, makes the contract unimplementable: nothing is left listening to honour it.
    let mut draining: Option<(Signal, Instant)> = None;
    let mut drain_tick = tokio::time::interval(DRAIN_POLL);
    drain_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            sig = shutdown.wait(), if draining.is_none() => {
                eprintln!("serve: draining websocket listener");
                // Stop the idle reaper: past this point every session is stopping anyway (both paths
                // make the same idempotent transition, so this is tidiness, not a race).
                if let Some(reaper) = &reaper {
                    reaper.abort();
                }
                // Best-effort remove the socket file so a restart isn't tripped by our own stale node.
                #[cfg(unix)]
                if let Some(path) = &uds_path {
                    let _ = std::fs::remove_file(path);
                }
                // New sessions refused and `/readyz` 503 from here on; sessions this replica already
                // owns keep running, and keep accepting reconnects, until the grace is spent.
                supervisor.begin_drain();
                draining = Some((sig, Instant::now() + cfg_drain_grace));
            }
            _ = drain_tick.tick(), if draining.is_some() => {
                // The `if` guard already proved this is `Some`; destructured rather than unwrapped so
                // the guard and the read cannot drift apart.
                if let Some((sig, deadline)) = draining
                    && (Instant::now() >= deadline || !supervisor.any_run_in_flight())
                {
                    // Stop every session (its retained input dropped, so it observes EOF and
                    // cancels+persists+exits), then wait for each task to be gone so persistence
                    // actually completes before the caller's `process::exit`.
                    supervisor.shutdown().await;
                    return Ok(Some(sig));
                }
            }
            Some(stream) = accept_tcp(&tcp_listener) => {
                let supervisor = supervisor.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_connection(&supervisor, stream).await {
                        eprintln!("serve: websocket connection error: {e}");
                    }
                });
            }
            Some(stream) = accept_uds_arm(&uds_listener) => {
                let supervisor = supervisor.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_connection(&supervisor, stream).await {
                        eprintln!("serve: unix socket connection error: {e}");
                    }
                });
            }
        }
    }
}

// The UDS counterpart to [`accept_tcp`], wrapped in a cfg-gated shim so the `select!` arm compiles
// identically on every target: on unix it awaits real `UnixStream`s (or `pending()`s when there's no
// listener); elsewhere it's a listener-less `pending()` forever (there's no UDS to bind, and
// `--listen-uds` errors before we ever reach here).
#[cfg(unix)]
async fn accept_uds_arm(listener: &Option<UnixListener>) -> Option<UnixStream> {
    match listener {
        Some(l) => match l.accept().await {
            Ok((stream, _addr)) => Some(stream),
            Err(e) => {
                eprintln!("serve: unix socket accept failed: {e}");
                None
            }
        },
        None => std::future::pending().await,
    }
}
#[cfg(not(unix))]
async fn accept_uds_arm(_listener: &Option<()>) -> Option<TcpStream> {
    std::future::pending().await
}

/// Read one HTTP/1.1 request and either attach it as a WebSocket or run it as a one-shot POST
/// command. Generic over the underlying byte stream so the same path serves both TCP and Unix-domain
/// sockets. We parse the request-line ourselves rather than handing every connection to tungstenite:
/// a POST has a body tungstenite's upgrade handshake would discard, and a GET that isn't an upgrade
/// deserves 405 rather than a failed handshake.
async fn handle_connection<S>(
    supervisor: &Arc<Supervisor>,
    mut stream: S,
) -> Result<(), Box<dyn std::error::Error>>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let (mut head, leftover) = match read_http_head(&mut stream).await {
        Ok(v) => v,
        Err(e) => {
            supervisor.refuse(&mut stream, &e).await;
            return Ok(());
        }
    };
    // Taken out of the head immediately, before anything else can read, log, or forward it: a grant
    // carries sealed credentials, and the upgrade's own headers are handed on to `create_response`.
    let grant = head.take_grant();

    // Health probes, before the path check and before any grant is demanded: `/livez` and `/readyz`
    // are the orchestrator's, not a tenant's. A probe that had to carry a session grant could never
    // be issued by the thing whose job is to decide whether this replica may have sessions at all.
    if matches!(head.path.as_str(), LIVEZ_PATH | READYZ_PATH) {
        // A non-GET probe is answered 405 without its body being parsed, so drain it for the same
        // reason the wrong-path 404 below does.
        if head.method != "GET" {
            drain_http_body(&mut stream, &head, &leftover).await;
        }
        let _ = handle_health(supervisor, &mut stream, &head).await;
        return Ok(());
    }

    if head.path != WS_PATH {
        refuse(&mut stream, &head, &leftover, HttpError::NotFound).await;
        return Ok(());
    }

    let requested_id = session_id_from_query(head.query.as_deref());
    if let Some(id) = &requested_id
        && !is_valid_session_id(id)
    {
        refuse(
            &mut stream,
            &head,
            &leftover,
            HttpError::BadRequest("invalid session_id"),
        )
        .await;
        return Ok(());
    }

    // Before the method branch, so the WebSocket and POST paths cannot diverge on who is allowed in.
    let service = match &supervisor.service {
        None => None,
        Some(svc) => match crate::service::authorize(
            &svc.verifier,
            &svc.shards,
            grant.as_deref(),
            requested_id.as_deref(),
            crate::service::now_unix(),
        ) {
            Ok(session) => Some(Arc::new(session)),
            Err(refusal) => {
                refuse(&mut stream, &head, &leftover, HttpError::from(refusal)).await;
                return Ok(());
            }
        },
    };

    match head.method.as_str() {
        "POST" => {
            if let Err(e) = handle_http_post(
                supervisor,
                &mut stream,
                &head,
                leftover,
                requested_id,
                service,
            )
            .await
            {
                supervisor.refuse(&mut stream, &e).await;
            }
            Ok(())
        }
        "GET" => {
            handle_websocket_upgrade(supervisor, stream, &head, leftover, requested_id, service)
                .await
        }
        _ => {
            refuse(&mut stream, &head, &leftover, HttpError::MethodNotAllowed).await;
            Ok(())
        }
    }
}

/// Answer `/livez` or `/readyz`.
///
/// Both exist on every `serve --listen`/`--listen-uds` daemon, in service mode or not, and neither
/// reads a grant — see the call site. Neither touches the session table or any tenant directory, so a
/// probe costs nothing a busy replica will notice.
///
/// - **`/livez` → 200** once the process is serving. Reaching this function *is* the proof: the
///   listener accepted a connection and the runtime read a request off it. Liveness deliberately
///   ignores the shards — a mount that has gone away is not fixed by killing the process, and a
///   liveness probe that failed on it would turn one bad mount into a restart loop.
/// - **`/readyz` → 200** when this replica can take a session: the listener is bound (again, proven
///   by the request), the grant verifier is loaded (a type-level invariant — `serve` refuses to start
///   `--service` without one, so `service.is_some()` *is* "verified keyring present"), and every
///   `--shard` is a directory this process can write to. Otherwise **503** with a one-line reason.
async fn handle_health<S: AsyncWrite + Unpin>(
    supervisor: &Arc<Supervisor>,
    stream: &mut S,
    head: &HttpHead,
) -> Result<(), HttpError> {
    if head.method != "GET" {
        return write_http_err(stream, &HttpError::MethodNotAllowed, None).await;
    }
    if head.path == LIVEZ_PATH {
        return write_http_ok(stream, 200, "OK", None, br#"{"status":"alive"}"#).await;
    }
    // A draining replica is not ready, whatever its mounts say, and it must say so on the very first
    // probe after SIGTERM — that answer is the entire mechanism by which the load balancer stops
    // sending it new sessions. Read before the probe, and out of the lock, so a slow mount cannot
    // delay the one answer that has to be immediate.
    let draining = lock_ignoring_poison(&supervisor.table).lifecycle != Lifecycle::Open;
    if draining {
        let body = json!({ "status": "not ready", "reason": "draining" }).to_string();
        return write_http_ok(stream, 503, "Service Unavailable", None, body.as_bytes()).await;
    }
    // No `--shard` means no service mode, and then "the listener is up" is the whole of readiness.
    let reason = match &supervisor.service {
        Some(svc) => {
            // Timed here rather than inside the probe: a cache hit is microseconds and a hit is most
            // of them, so the histogram's shape is exactly "how often did a caller have to wait for
            // the mount, and how long". That is the question a degrading mount is answered by.
            let began = std::time::Instant::now();
            let answer = svc.ready.check(&svc.shards).await;
            if let Some(m) = &supervisor.metrics {
                m.ready_probe_seconds.observe(began.elapsed().as_secs_f64());
            }
            answer
        }
        None => None,
    };
    match reason {
        None => write_http_ok(stream, 200, "OK", None, br#"{"status":"ready"}"#).await,
        Some(reason) => {
            let body = json!({ "status": "not ready", "reason": reason }).to_string();
            write_http_ok(stream, 503, "Service Unavailable", None, body.as_bytes()).await
        }
    }
}

/// The answer a probe produces: `None` is ready, `Some(reason)` is not.
type ReadyAnswer = Option<String>;

/// What [`ReadyCache`] holds between probes: the last answer, and the probe currently running.
#[derive(Default)]
struct ReadyState {
    /// The last completed answer and when it landed, reused for [`READY_CACHE_TTL`].
    cached: Option<(Instant, ReadyAnswer)>,
    /// The probe in flight, if one is. `None` in the watched value means "still running"; every
    /// caller that arrives while it is set waits on this instead of starting a probe of its own.
    inflight: Option<tokio::sync::watch::Receiver<Option<ReadyAnswer>>>,
}

/// The `/readyz` shard probe: **single-flight**, memoized for [`READY_CACHE_TTL`], and bounded by
/// [`READY_PROBE_TIMEOUT`]. `None` means ready.
///
/// Both properties exist for the same failure. `/readyz` is unauthenticated by design (an
/// orchestrator deciding whether this replica may hold sessions has no session grant and could never
/// obtain one — see [`handle_connection`]), and its probe is filesystem I/O on a hard-mounted network
/// filesystem, where "unreachable" is an uninterruptible sleep rather than an error. Writing the memo
/// only *after* the probe returned meant every request arriving during a probe missed the cache and
/// spawned another one; `spawn_blocking` tasks are uncancellable, so each of those permanently
/// consumed a blocking-pool thread. A 10-second `HEALTHCHECK` plus a load-balancer probe then leaked
/// roughly a thread every 10 seconds into a pool that caps at 512, after which *every* `spawn_blocking`
/// in the process — session listing included — queued behind dead threads, while `/livez` (deliberately
/// unconditional) kept the replica alive.
///
/// So: at most one probe runs at a time and every waiter shares its result, and a waiter that hits the
/// deadline answers "not ready" **without** starting a second probe. The timed-out probe's thread is
/// still gone — nothing can reclaim it — but it is one thread per hung mount rather than one per
/// request, and its eventual answer still lands in the memo for whoever asks next.
#[derive(Default)]
struct ReadyCache(Arc<Mutex<ReadyState>>);

impl ReadyCache {
    async fn check(&self, shards: &Arc<Shards>) -> ReadyAnswer {
        let shards = shards.clone();
        self.check_with(move || probe_shards(&shards), READY_PROBE_TIMEOUT)
            .await
    }

    /// [`check`](Self::check) with the probe and the deadline injected, so a test can drive both.
    async fn check_with<F>(&self, probe: F, deadline: Duration) -> ReadyAnswer
    where
        F: FnOnce() -> ReadyAnswer + Send + 'static,
    {
        let mut rx = {
            let mut state = lock_ignoring_poison(&self.0);
            if let Some((at, reason)) = &state.cached
                && at.elapsed() < READY_CACHE_TTL
            {
                return reason.clone();
            }
            match &state.inflight {
                Some(rx) => rx.clone(),
                None => {
                    let (tx, rx) = tokio::sync::watch::channel(None);
                    state.inflight = Some(rx.clone());
                    let shared = Arc::clone(&self.0);
                    // The join lives on a task of its own rather than on the caller, so *every*
                    // caller — the one that started this probe included — waits on the watch under
                    // the same deadline, and a caller that gives up leaves the probe running for the
                    // next one rather than abandoning its result.
                    //
                    // Off the runtime thread, as before: the process runs a single-threaded runtime,
                    // and this is a `stat` plus a create/unlink on a network filesystem — exactly the
                    // I/O that stalls when an EFS mount hiccups. Blocking here would wedge every
                    // session on the replica on behalf of a probe.
                    tokio::spawn(async move {
                        let reason = tokio::task::spawn_blocking(probe)
                            .await
                            .unwrap_or_else(|_| Some("shard probe did not complete".to_string()));
                        // Memo and in-flight slot move together under one lock, so a caller can never
                        // observe "no probe running" alongside a stale answer and start a second one.
                        {
                            let mut state = lock_ignoring_poison(&shared);
                            state.cached = Some((Instant::now(), reason.clone()));
                            state.inflight = None;
                        }
                        let _ = tx.send(Some(reason));
                    });
                    rx
                }
            }
        };
        match tokio::time::timeout(deadline, rx.wait_for(Option::is_some)).await {
            Ok(Ok(answer)) => answer.clone().unwrap_or(None),
            // The probe task was dropped before it answered (the runtime is shutting down). Not
            // ready, and nothing to memoize.
            Ok(Err(_)) => Some("shard probe did not complete".to_string()),
            // Deliberately *not* memoized: the probe is still running and its real answer will land.
            Err(_) => Some("shard probe timed out".to_string()),
        }
    }
}

/// `None` if every mount is a directory this process can write to; otherwise the first failure.
///
/// The reason names the **shard**, never its path. A readiness body is readable by anything that can
/// reach the port, and the replica's mount layout is not a caller's business — the shard name is
/// already public (it prefixes every session id), and it is the only half an operator needs to know
/// which mount to look at.
fn probe_shards(shards: &Shards) -> Option<String> {
    shards.iter().find_map(|(name, path)| {
        probe_shard(path)
            .err()
            .map(|why| format!("shard {name}: {why}"))
    })
}

fn probe_shard(path: &std::path::Path) -> Result<(), &'static str> {
    match std::fs::metadata(path) {
        Err(_) => return Err("not mounted"),
        Ok(meta) if !meta.is_dir() => return Err("not a directory"),
        Ok(_) => {}
    }
    // A permission-bit check (`access(W_OK)`) would be one syscall, but it answers about the bits,
    // and the failures that actually happen here — a read-only remount, a full or unreachable mount —
    // only surface on a real write. So write. The name carries the pid so two replicas sharing a
    // mount cannot unlink each other's probe, and it is removed immediately either way.
    let probe = path.join(format!(".readyz.{}", std::process::id()));
    match std::fs::File::create(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            Ok(())
        }
        Err(_) => Err("not writable"),
    }
}

/// Finish the WebSocket handshake from an already-parsed GET and attach the socket to its session.
/// `leftover` is any bytes read past the header block (should be empty for a well-formed upgrade;
/// tungstenite treats them as the start of the WebSocket stream).
async fn handle_websocket_upgrade<S>(
    supervisor: &Arc<Supervisor>,
    mut stream: S,
    head: &HttpHead,
    leftover: Vec<u8>,
    requested_id: Option<String>,
    service: Option<Arc<ServiceSession>>,
) -> Result<(), Box<dyn std::error::Error>>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let request = match http_request_from_head(head) {
        Ok(r) => r,
        Err(e) => {
            supervisor.refuse(&mut stream, &e).await;
            return Ok(());
        }
    };
    // Pin **before** writing the 101: the session slot is where the tenant check and the
    // still-stopping wait live, and both of their answers are HTTP statuses (403, 503). Past the
    // upgrade the only vocabulary left is a close code.
    let pinned = match supervisor.pin(requested_id, service.clone()).await {
        Ok(pinned) => pinned,
        Err(e) => {
            supervisor.refuse(&mut stream, &HttpError::from(e)).await;
            return Ok(());
        }
    };
    let response = match create_response(&request) {
        Ok(r) => r,
        Err(_) => {
            supervisor.unpin(&pinned.id, pinned.incarnation);
            supervisor
                .refuse(&mut stream, &HttpError::UpgradeRequired)
                .await;
            return Ok(());
        }
    };
    if let Err(e) = write_raw_http_response(&mut stream, &response).await {
        supervisor.unpin(&pinned.id, pinned.incarnation);
        return Err(e.into());
    }

    let config = WebSocketConfig::default()
        .max_message_size(Some(MAX_INBOUND_MESSAGE_BYTES))
        .max_frame_size(Some(MAX_INBOUND_MESSAGE_BYTES));
    let ws =
        WebSocketStream::from_partially_read(stream, leftover, Role::Server, Some(config)).await;
    supervisor.attach(pinned, service, ws).await;
    Ok(())
}

/// POST `/_beyond/agent`: inject one command into the session and return its `ack` or `response`.
async fn handle_http_post<S>(
    supervisor: &Arc<Supervisor>,
    stream: &mut S,
    head: &HttpHead,
    leftover: Vec<u8>,
    requested_id: Option<String>,
    service: Option<Arc<ServiceSession>>,
) -> Result<(), HttpError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    if head
        .transfer_encoding
        .as_deref()
        .is_some_and(|t| !t.eq_ignore_ascii_case("identity"))
    {
        return Err(HttpError::LengthRequired);
    }
    let content_length = head.content_length.ok_or(HttpError::LengthRequired)?;
    let body = read_http_body(stream, &leftover, content_length).await?;
    if body.is_empty() {
        return Err(HttpError::BadRequest("empty body"));
    }

    let mut cmd: Value =
        serde_json::from_slice(&body).map_err(|_| HttpError::BadRequest("body is not JSON"))?;
    if !cmd.is_object() {
        return Err(HttpError::BadRequest("body must be a JSON object"));
    }
    // Always correlate by `id`: a POST sink also sees live frames from any concurrent WebSocket on
    // the same session, and without an id the first `response` of the matching command type would
    // be stolen. A client that omitted one still gets it echoed on the frame they receive.
    let client_id = match cmd.get("id").and_then(Value::as_str) {
        Some(id) if !id.is_empty() => id.to_string(),
        _ => {
            let minted = new_id();
            if let Value::Object(map) = &mut cmd {
                map.insert("id".into(), json!(minted.clone()));
            }
            minted
        }
    };
    let command = cmd
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if command.is_empty() {
        return Err(HttpError::BadRequest("missing `type`"));
    }

    if command == "list_daemon_sessions" {
        let frame = supervisor
            .list_daemon_sessions(Some(client_id.clone()), service.as_deref())
            .await;
        let Some(line) = frame_to_line(frame) else {
            return Err(HttpError::BadRequest("failed to serialize response"));
        };
        // No session was pinned — there isn't one to name. An empty header would be a lie; omit it.
        return write_http_ok(stream, 200, "OK", None, &line).await;
    }

    let line = serde_json::to_string(&cmd)
        .map_err(|_| HttpError::BadRequest("failed to serialize command"))?;

    let Pinned {
        id,
        incarnation,
        input_tx,
        out_conn,
    } = supervisor
        .pin(requested_id, service)
        .await
        .map_err(HttpError::from)?;
    let (conn_tx, mut conn_rx) = mpsc::channel::<OutFrame>(OUT_CHANNEL_BOUND);
    // No catch-up: a POST is one command's reply, not a streaming attach. Seeding history here
    // would dump the transcript into a buffer the waiter has to skip, and could fill it before the
    // ack ever arrived.
    let sink_id = lock_ignoring_poison(&out_conn).add(OutSink::Bounded(conn_tx));

    // Unpin + remove the sink on every exit (timeout, send failure, reply). A session with
    // `attached > 0` is invisible to the idle reaper; leaking a pin would pin it for the daemon's
    // life.
    struct PinGuard {
        supervisor: Arc<Supervisor>,
        id: String,
        incarnation: u64,
        out_conn: SharedOutConn,
        sink_id: u64,
    }
    impl Drop for PinGuard {
        fn drop(&mut self) {
            lock_ignoring_poison(&self.out_conn).remove(self.sink_id);
            self.supervisor.unpin(&self.id, self.incarnation);
        }
    }
    let _guard = PinGuard {
        supervisor: Arc::clone(supervisor),
        id: id.clone(),
        incarnation,
        out_conn,
        sink_id,
    };

    if input_tx.send(line).await.is_err() {
        return Err(HttpError::BadRequest("session ended"));
    }

    let wait = async {
        loop {
            match conn_rx.recv().await {
                Some(frame) => {
                    if let Some(v) = frame_as_value(&frame)
                        && reply_matches(&v, &command, &client_id)
                    {
                        return Ok((v["type"].as_str() == Some("ack"), frame));
                    }
                }
                None => return Err(HttpError::BadRequest("session ended")),
            }
        }
    };
    let (is_ack, frame) = match tokio::time::timeout(HTTP_REPLY_TIMEOUT, wait).await {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err(HttpError::Timeout),
    };
    let Some(body) = frame_to_line(frame) else {
        return Err(HttpError::BadRequest("failed to serialize response"));
    };
    let (status, reason) = if is_ack && command == "prompt" {
        (202, "Accepted")
    } else {
        (200, "OK")
    };
    write_http_ok(stream, status, reason, Some(&id), &body).await
}

fn frame_as_value(frame: &OutFrame) -> Option<Value> {
    match frame {
        OutFrame::Value(v) => Some(v.clone()),
        OutFrame::Raw(b) => serde_json::from_slice(b).ok(),
    }
}

/// Whether `v` is the `ack` or `response` this POST is waiting on. Matched on `id` (always set by
/// [`handle_http_post`]) so a concurrent WebSocket's frames on the same session cannot satisfy it.
fn reply_matches(v: &Value, command: &str, client_id: &str) -> bool {
    let ty = v.get("type").and_then(Value::as_str);
    if ty != Some("ack") && ty != Some("response") {
        return false;
    }
    v.get("id").and_then(Value::as_str) == Some(client_id)
        && v.get("command").and_then(Value::as_str) == Some(command)
}

#[derive(Debug)]
struct HttpHead {
    method: String,
    path: String,
    query: Option<String>,
    path_and_query: String,
    content_length: Option<usize>,
    transfer_encoding: Option<String>,
    headers: Vec<(String, String)>,
}

impl HttpHead {
    /// Remove and return the `x-beyond-grant` header.
    ///
    /// **Removing** it is the point. A grant carries sealed credentials, and this head is handed to
    /// `create_response`, which echoes the request's headers into the handshake — so a grant left in
    /// place would be reflected straight back to the client, and would sit in the `Debug` of any
    /// head logged on an error path. Read once, here, and gone.
    ///
    /// A value over [`MAX_GRANT_BYTES`](crate::service::MAX_GRANT_BYTES) is dropped rather than
    /// returned: it cannot be a grant this fleet minted, and there is no reason to run crypto over
    /// it. The caller then sees "no grant" — a 401, which is what an unusable token deserves.
    fn take_grant(&mut self) -> Option<String> {
        let idx = self
            .headers
            .iter()
            .position(|(name, _)| name.eq_ignore_ascii_case(crate::service::GRANT_HEADER))?;
        let (_, value) = self.headers.remove(idx);
        (value.len() <= crate::service::MAX_GRANT_BYTES).then_some(value)
    }
}

fn session_id_from_query(query: Option<&str>) -> Option<String> {
    let query = query?;
    url::form_urlencoded::parse(query.as_bytes())
        .find_map(|(k, v)| (k == "session_id").then(|| v.into_owned()))
}

#[derive(Debug)]
enum HttpError {
    Incomplete,
    BadRequest(&'static str),
    NotFound,
    MethodNotAllowed,
    UpgradeRequired,
    PayloadTooLarge,
    LengthRequired,
    Timeout,
    /// The supervisor won't attach right now ([`Supervisor::pin`]): the daemon is shutting down, or the
    /// id's previous session task is still exiting. Retryable.
    Unavailable(&'static str),
    /// Service mode: no session grant, or one that doesn't verify.
    Unauthorized(&'static str),
    /// Service mode: the grant verifies but isn't for this session or this tenant.
    Forbidden(&'static str),
    /// Service mode: the grant is well-formed, but this replica doesn't mount its shard. The client
    /// should be routed elsewhere rather than retried here.
    Misdirected(&'static str),
    Io,
}

impl From<Refusal> for HttpError {
    fn from(r: Refusal) -> Self {
        match r {
            Refusal::Unauthorized(m) => HttpError::Unauthorized(m),
            Refusal::BadRequest(m) => HttpError::BadRequest(m),
            Refusal::Forbidden(m) => HttpError::Forbidden(m),
            Refusal::Misdirected(m) => HttpError::Misdirected(m),
        }
    }
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpError::Incomplete => write!(f, "incomplete request"),
            HttpError::BadRequest(m)
            | HttpError::Unavailable(m)
            | HttpError::Unauthorized(m)
            | HttpError::Forbidden(m)
            | HttpError::Misdirected(m) => write!(f, "{m}"),
            HttpError::NotFound => write!(f, "not found: expected {WS_PATH}"),
            HttpError::MethodNotAllowed => write!(f, "method not allowed"),
            HttpError::UpgradeRequired => write!(f, "WebSocket upgrade required"),
            HttpError::PayloadTooLarge => write!(f, "payload too large"),
            HttpError::LengthRequired => write!(f, "Content-Length required"),
            HttpError::Timeout => write!(f, "timed out waiting for session reply"),
            HttpError::Io => write!(f, "i/o error"),
        }
    }
}

impl HttpError {
    fn status(&self) -> (u16, &'static str) {
        match self {
            HttpError::NotFound => (404, "Not Found"),
            HttpError::MethodNotAllowed => (405, "Method Not Allowed"),
            HttpError::UpgradeRequired => (426, "Upgrade Required"),
            HttpError::PayloadTooLarge => (413, "Payload Too Large"),
            HttpError::LengthRequired => (411, "Length Required"),
            HttpError::Timeout => (504, "Gateway Timeout"),
            HttpError::Unavailable(_) => (503, "Service Unavailable"),
            HttpError::Unauthorized(_) => (401, "Unauthorized"),
            HttpError::Forbidden(_) => (403, "Forbidden"),
            HttpError::Misdirected(_) => (421, "Misdirected Request"),
            HttpError::Incomplete | HttpError::BadRequest(_) | HttpError::Io => {
                (400, "Bad Request")
            }
        }
    }
}

/// Parse a complete HTTP/1.1 header block. `Incomplete` means the buffer does not yet contain
/// `\r\n\r\n` (or httparse still wants more) — the reader should append and retry.
fn parse_http_head(buf: &[u8]) -> Result<(HttpHead, usize), HttpError> {
    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut req = httparse::Request::new(&mut headers);
    let header_len = match req.parse(buf) {
        Ok(httparse::Status::Complete(n)) => n,
        Ok(httparse::Status::Partial) => return Err(HttpError::Incomplete),
        Err(_) => return Err(HttpError::BadRequest("malformed request")),
    };
    let method = req
        .method
        .ok_or(HttpError::BadRequest("missing method"))?
        .to_string();
    let path_and_query = req
        .path
        .ok_or(HttpError::BadRequest("missing path"))?
        .to_string();
    let (path, query) = match path_and_query.split_once('?') {
        Some((p, q)) => (p.to_string(), Some(q.to_string())),
        None => (path_and_query.clone(), None),
    };
    let mut content_length = None;
    let mut transfer_encoding = None;
    let mut collected = Vec::with_capacity(req.headers.len());
    for h in req.headers {
        let name = h.name;
        let value = std::str::from_utf8(h.value).unwrap_or("");
        if name.eq_ignore_ascii_case("content-length") {
            let n: usize = value
                .trim()
                .parse()
                .map_err(|_| HttpError::BadRequest("invalid Content-Length"))?;
            if n > MAX_INBOUND_MESSAGE_BYTES {
                return Err(HttpError::PayloadTooLarge);
            }
            content_length = Some(n);
        }
        if name.eq_ignore_ascii_case("transfer-encoding") {
            transfer_encoding = Some(value.to_string());
        }
        collected.push((name.to_string(), value.to_string()));
    }
    Ok((
        HttpHead {
            method,
            path,
            query,
            path_and_query,
            content_length,
            transfer_encoding,
            headers: collected,
        },
        header_len,
    ))
}

async fn read_http_head<S: AsyncRead + Unpin>(
    stream: &mut S,
) -> Result<(HttpHead, Vec<u8>), HttpError> {
    let mut buf = Vec::with_capacity(512);
    let mut tmp = [0u8; 1024];
    loop {
        let n = stream.read(&mut tmp).await.map_err(|_| HttpError::Io)?;
        if n == 0 {
            return Err(if buf.is_empty() {
                HttpError::BadRequest("empty request")
            } else {
                HttpError::BadRequest("truncated request")
            });
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.len() > MAX_HEADER_BYTES {
            return Err(HttpError::BadRequest("headers too large"));
        }
        match parse_http_head(&buf) {
            Ok((head, header_len)) => {
                let leftover = buf[header_len..].to_vec();
                return Ok((head, leftover));
            }
            Err(HttpError::Incomplete) => continue,
            Err(e) => return Err(e),
        }
    }
}

async fn read_http_body<S: AsyncRead + Unpin>(
    stream: &mut S,
    leftover: &[u8],
    content_length: usize,
) -> Result<Vec<u8>, HttpError> {
    if content_length > MAX_INBOUND_MESSAGE_BYTES {
        return Err(HttpError::PayloadTooLarge);
    }
    let mut body = Vec::with_capacity(content_length);
    let take = leftover.len().min(content_length);
    body.extend_from_slice(&leftover[..take]);
    while body.len() < content_length {
        let mut tmp = [0u8; 8192];
        let want = (content_length - body.len()).min(tmp.len());
        let n = stream
            .read(&mut tmp[..want])
            .await
            .map_err(|_| HttpError::Io)?;
        if n == 0 {
            return Err(HttpError::BadRequest("truncated body"));
        }
        body.extend_from_slice(&tmp[..n]);
    }
    Ok(body)
}

fn http_request_from_head(head: &HttpHead) -> Result<http::Request<()>, HttpError> {
    let mut builder = http::Request::builder()
        .method(head.method.as_str())
        .uri(head.path_and_query.as_str());
    for (k, v) in &head.headers {
        builder = builder.header(k.as_str(), v.as_str());
    }
    builder
        .body(())
        .map_err(|_| HttpError::BadRequest("malformed request"))
}

async fn write_raw_http_response<S: AsyncWrite + Unpin>(
    stream: &mut S,
    response: &http::Response<()>,
) -> std::io::Result<()> {
    let status = response.status();
    let reason = status.canonical_reason().unwrap_or("OK");
    let mut out = format!("HTTP/1.1 {} {reason}\r\n", status.as_u16());
    for (k, v) in response.headers() {
        out.push_str(k.as_str());
        out.push_str(": ");
        if let Ok(v) = v.to_str() {
            out.push_str(v);
        }
        out.push_str("\r\n");
    }
    out.push_str("\r\n");
    stream.write_all(out.as_bytes()).await?;
    stream.flush().await
}

async fn write_http_ok<S: AsyncWrite + Unpin>(
    stream: &mut S,
    status: u16,
    reason: &str,
    session_id: Option<&str>,
    body: &[u8],
) -> Result<(), HttpError> {
    let mut out = format!("HTTP/1.1 {status} {reason}\r\n");
    out.push_str("Content-Type: application/json\r\n");
    out.push_str(&format!("Content-Length: {}\r\n", body.len()));
    out.push_str("Connection: close\r\n");
    if let Some(id) = session_id {
        out.push_str(SESSION_ID_HEADER);
        out.push_str(": ");
        out.push_str(id);
        out.push_str("\r\n");
    }
    if status == 405 {
        out.push_str("Allow: GET, POST\r\n");
    }
    if status == 426 {
        out.push_str("Upgrade: websocket\r\n");
    }
    // Every 503 this server sends is retryable and short-lived — a session still stopping, one open
    // on another replica, or a replica at its live-session cap — so say so in the one header a proxy,
    // a client library or a retry policy already knows how to read.
    if status == 503 {
        out.push_str("Retry-After: 1\r\n");
    }
    out.push_str("\r\n");
    stream
        .write_all(out.as_bytes())
        .await
        .map_err(|_| HttpError::Io)?;
    stream.write_all(body).await.map_err(|_| HttpError::Io)?;
    stream.flush().await.map_err(|_| HttpError::Io)
}

/// Answer a request this connection refuses **before** parsing its body: drain the body, then write
/// the error. Every early refusal goes through this, so none of them can be the one that forgets
/// ([`drain_http_body`] explains what forgetting costs).
async fn refuse<S: AsyncRead + AsyncWrite + Unpin>(
    stream: &mut S,
    head: &HttpHead,
    leftover: &[u8],
    err: HttpError,
) {
    drain_http_body(stream, head, leftover).await;
    let _ = write_http_err(stream, &err, None).await;
}

/// Read and discard a request body this connection is never going to parse, before answering with an
/// error and closing.
///
/// Closing a socket that still holds unread bytes makes the kernel send RST instead of FIN, and an RST
/// **discards data already queued for the peer** — including the error response just written. A client
/// that POSTs to the wrong path would see `ECONNRESET` rather than the `404` explaining what it got
/// wrong, and would have no way to tell that from the replica dying mid-request.
///
/// Bounded by [`MAX_INBOUND_MESSAGE_BYTES`]: past that the body is not worth reading to be polite
/// about, and the oversize case already has its own answer ([`HttpError::PayloadTooLarge`]).
async fn drain_http_body<S: AsyncRead + Unpin>(stream: &mut S, head: &HttpHead, leftover: &[u8]) {
    let Some(len) = head.content_length else {
        return;
    };
    if len > MAX_INBOUND_MESSAGE_BYTES {
        return;
    }
    let mut left = len.saturating_sub(leftover.len());
    let mut tmp = [0u8; 8192];
    while left > 0 {
        let want = left.min(tmp.len());
        match stream.read(&mut tmp[..want]).await {
            Ok(0) | Err(_) => return,
            Ok(n) => left -= n,
        }
    }
}

async fn write_http_err<S: AsyncWrite + Unpin>(
    stream: &mut S,
    err: &HttpError,
    session_id: Option<&str>,
) -> Result<(), HttpError> {
    let (status, reason) = err.status();
    let body = json!({ "error": err.to_string() }).to_string();
    write_http_ok(stream, status, reason, session_id, body.as_bytes()).await
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// Build a `Live` handle with no session behind it. The input receiver is returned and must be held
    /// to keep the session "alive"; dropping it is how a test models a session whose loop ended.
    fn handle(
        attached: usize,
        detached_ago: Option<Duration>,
    ) -> (SessionHandle, mpsc::Receiver<String>) {
        let (input_tx, input_rx) = mpsc::channel::<String>(crate::serve::IN_CHANNEL_BOUND);
        let h = SessionHandle {
            incarnation: 0,
            phase: Phase::Live(input_tx),
            tenant: None,
            out_conn: Arc::new(Mutex::new(OutFanout::default())),
            exited: CancellationToken::new(),
            attached,
            last_detached_at: detached_ago.and_then(|d| Instant::now().checked_sub(d)),
            running: Arc::new(AtomicBool::new(false)),
            end_reason: Arc::new(Mutex::new(crate::metrics::SessionEnd::Client)),
        };
        (h, input_rx)
    }

    #[test]
    fn idle_timeout_defaults_to_a_finite_window_and_zero_opts_out() {
        assert_eq!(
            resolve_idle_timeout(None, false),
            Some(DEFAULT_SESSION_IDLE_TIMEOUT),
            "no --session-idle-timeout must still reap: the map has no other way to shrink"
        );
        assert_eq!(
            resolve_idle_timeout(Some(Duration::ZERO), false),
            None,
            "0 opts out"
        );
        assert_eq!(
            resolve_idle_timeout(Some(Duration::from_secs(5)), false),
            Some(Duration::from_secs(5))
        );
    }

    #[test]
    fn service_mode_holds_a_detached_session_for_a_minute_not_an_hour() {
        // The hour is a single-user daemon's number: there, re-attaching means reconnecting to *this
        // process*. On a replica the session is on shared storage and any replica respawns it, so
        // what the window really governs is how long a detached session keeps its lock — and an hour
        // of that is how a failed-over session stays unreachable from the replica the edge's hash
        // chooses.
        assert_eq!(
            resolve_idle_timeout(None, true),
            Some(DEFAULT_SERVICE_SESSION_IDLE_TIMEOUT)
        );
        assert!(
            DEFAULT_SERVICE_SESSION_IDLE_TIMEOUT < DEFAULT_SESSION_IDLE_TIMEOUT,
            "the replica's window must be the shorter one"
        );

        // The mode only decides what *silence* means. An operator who names a window gets it, and
        // `0` still pins every session, in either mode.
        assert_eq!(
            resolve_idle_timeout(Some(Duration::from_secs(5)), true),
            Some(Duration::from_secs(5))
        );
        assert_eq!(resolve_idle_timeout(Some(Duration::ZERO), true), None);
    }

    #[test]
    fn a_dead_session_is_reapable_however_it_looks_otherwise() {
        // Its loop ended (input receiver gone) while a connection is *still attached* and its idle clock
        // never started: every ordinary condition says "keep", and it must still be stopped — it is no
        // longer attachable in any useful sense, and its task is on its way out.
        let (h, input_rx) = handle(1, None);
        drop(input_rx);
        assert!(is_reapable(&h, Duration::from_secs(3600)));
    }

    #[test]
    fn a_live_session_is_reapable_only_once_detached_past_the_timeout() {
        let timeout = Duration::from_secs(60);

        let (attached, _rx) = handle(1, None);
        assert!(!is_reapable(&attached, timeout), "attached: never");

        let (fresh, _rx) = handle(0, None);
        assert!(
            !is_reapable(&fresh, timeout),
            "detached but no clock started: never"
        );

        let (recent, _rx) = handle(0, Some(Duration::from_secs(10)));
        assert!(
            !is_reapable(&recent, timeout),
            "detached, but not past the timeout"
        );

        let (idle, _rx) = handle(0, Some(Duration::from_secs(120)));
        assert!(is_reapable(&idle, timeout), "detached past the timeout");

        let (mid_run, _rx) = handle(0, Some(Duration::from_secs(120)));
        mid_run.running.store(true, Ordering::Relaxed);
        assert!(
            !is_reapable(&mid_run, timeout),
            "a detached background run must never be reaped out from under its turn"
        );

        let (mut stopping, _rx) = handle(0, Some(Duration::from_secs(120)));
        stopping.stop(crate::metrics::SessionEnd::Client);
        assert!(
            !is_reapable(&stopping, timeout),
            "already on its way out: nothing left to reap"
        );
    }

    /// A wedged session must not hold the caller past the grace period.
    #[tokio::test]
    async fn a_wedged_task_does_not_hold_the_join_past_the_grace_period() {
        let wedged = CancellationToken::new();
        let finished = CancellationToken::new();
        finished.cancel();

        let start = Instant::now();
        await_exits_within(vec![finished, wedged], Duration::from_millis(200)).await;
        let waited = start.elapsed();

        assert!(
            waited >= Duration::from_millis(200) && waited < Duration::from_secs(5),
            "the wait must end at the grace period, not when the wedged session finally exits: {waited:?}"
        );
    }

    #[tokio::test]
    async fn joining_returns_as_soon_as_every_task_has_exited() {
        let exits: Vec<_> = (0..4)
            .map(|_| {
                let e = CancellationToken::new();
                e.cancel();
                e
            })
            .collect();
        let start = Instant::now();
        await_exits_within(exits, Duration::from_secs(10)).await;
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "finished tasks must not wait out the grace period"
        );
    }

    /// Stands in for `serve_session` so a test decides when a session finishes exiting. It counts how
    /// many bodies are running at once; once its loop ends it parks — "still persisting" — until
    /// `release` fires.
    #[derive(Default)]
    struct Probe {
        running_now: std::sync::atomic::AtomicUsize,
        peak: std::sync::atomic::AtomicUsize,
        started: std::sync::atomic::AtomicUsize,
        exiting: std::sync::atomic::AtomicUsize,
        release: CancellationToken,
    }

    impl Probe {
        fn started(&self) -> usize {
            self.started.load(Ordering::SeqCst)
        }
        fn exiting(&self) -> usize {
            self.exiting.load(Ordering::SeqCst)
        }
        fn peak(&self) -> usize {
            self.peak.load(Ordering::SeqCst)
        }
    }

    /// A supervisor whose sessions are [`Probe`] bodies. Each runs until its input closes (reaped, shut
    /// down) — or, for the very first body when `first_ends_itself`, drops its input at once, as a loop
    /// that failed on its own does — then parks until the probe's `release`.
    fn probe_supervisor(probe: &Arc<Probe>, first_ends_itself: bool) -> Arc<Supervisor> {
        let probe = probe.clone();
        Arc::new(Supervisor {
            metrics: None,
            table: Arc::default(),
            session_dir: None,
            service: None,
            body: Box::new(move |_id, _service, mut input_rx, _out, _running| {
                let probe = probe.clone();
                Box::pin(async move {
                    let now = probe.running_now.fetch_add(1, Ordering::SeqCst) + 1;
                    probe.peak.fetch_max(now, Ordering::SeqCst);
                    let nth = probe.started.fetch_add(1, Ordering::SeqCst);
                    if first_ends_itself && nth == 0 {
                        drop(input_rx);
                    } else {
                        while input_rx.recv().await.is_some() {}
                    }
                    probe.exiting.fetch_add(1, Ordering::SeqCst);
                    probe.release.cancelled().await;
                    probe.running_now.fetch_sub(1, Ordering::SeqCst);
                })
            }),
        })
    }

    /// Yield (on the test's current-thread runtime, so every other ready task runs) until `cond` holds.
    async fn until(what: &str, cond: impl Fn() -> bool) {
        for _ in 0..1000 {
            if cond() {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("never happened: {what}");
    }

    /// Give every ready task ample turns to do whatever it is going to do.
    async fn settle() {
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
    }

    /// Attach to `id`, then detach, leaving an idle session a zero-timeout reap takes at once.
    async fn attach_and_detach(sup: &Supervisor, id: &str) -> u64 {
        let p = sup.pin(Some(id.into()), None).await.unwrap();
        sup.unpin(&p.id, p.incarnation);
        p.incarnation
    }

    /// The regression: a reconnect landing between a reap and the reaped task's exit used to find the
    /// id free and spawn a second `serve_session` on the same file while the first was still
    /// persisting. It must wait for the first to be gone.
    #[tokio::test]
    async fn a_reconnect_during_a_reap_waits_for_the_old_task_to_exit() {
        let probe = Arc::new(Probe::default());
        let sup = probe_supervisor(&probe, false);
        let first = attach_and_detach(&sup, "s1").await;
        until("the first session starts", || probe.started() == 1).await;

        // Reap it. Its input closes and it parks mid-exit: the old task is still persisting.
        sup.reap_idle(Duration::ZERO);
        until("the reaped session sees EOF", || probe.exiting() == 1).await;

        let reconnect = tokio::spawn({
            let sup = sup.clone();
            async move { sup.pin(Some("s1".into()), None).await }
        });
        settle().await;
        assert_eq!(
            probe.started(),
            1,
            "a reconnect during the reap must not start a second session while the first is exiting"
        );
        assert!(!reconnect.is_finished(), "the reconnect waits");

        // The old task finishes and frees the id: only now does the reconnect get a fresh session.
        probe.release.cancel();
        let second = reconnect.await.unwrap().unwrap();
        until("the second session starts", || probe.started() == 2).await;
        assert!(second.incarnation > first);
        assert_eq!(probe.peak(), 1, "never two session tasks on one id at once");
    }

    /// Same shape, but the old task never finishes: the reconnect gives up — refused, not hung, and
    /// still never alongside it.
    #[tokio::test]
    async fn a_reconnect_behind_a_wedged_exit_is_refused_not_doubled() {
        let probe = Arc::new(Probe::default());
        let sup = probe_supervisor(&probe, false);
        attach_and_detach(&sup, "s1").await;
        until("the session starts", || probe.started() == 1).await;
        sup.reap_idle(Duration::ZERO);
        until("the reaped session sees EOF", || probe.exiting() == 1).await;

        let refused = sup
            .pin_within(Some("s1".into()), None, Duration::from_millis(50))
            .await;
        assert!(refused.is_err(), "a wedged predecessor refuses the attach");
        assert_eq!(probe.started(), 1, "and never starts a rival");
        probe.release.cancel();
    }

    /// A session whose loop ended on its own (closed input) but whose task hasn't exited yet is
    /// attachable in name only: a reconnect waits for it rather than respawning beside it. And the
    /// first connection's late unpin must not detach the incarnation that replaced it.
    #[tokio::test]
    async fn a_reconnect_to_a_session_whose_loop_ended_waits_for_its_task() {
        let probe = Arc::new(Probe::default());
        let sup = probe_supervisor(&probe, true);
        let first = sup.pin(Some("s1".into()), None).await.unwrap();
        until("the loop ends", || probe.exiting() == 1).await;
        assert!(first.input_tx.is_closed());

        let reconnect = tokio::spawn({
            let sup = sup.clone();
            async move { sup.pin(Some("s1".into()), None).await }
        });
        settle().await;
        assert_eq!(probe.started(), 1, "no respawn beside a task still exiting");
        assert!(!reconnect.is_finished());

        probe.release.cancel();
        let second = reconnect.await.unwrap().unwrap();
        until("the second session starts", || probe.started() == 2).await;
        assert_eq!(probe.peak(), 1);

        // Were the stale unpin counted, the second session would read as detached and be reaped with a
        // connection still attached.
        sup.unpin(&first.id, first.incarnation);
        sup.reap_idle(Duration::ZERO);
        let table = lock_ignoring_poison(&sup.table);
        let h = &table.sessions["s1"];
        assert_eq!(h.incarnation, second.incarnation);
        assert_eq!(h.attached, 1);
        assert!(h.phase.input().is_some(), "still attached, so not reaped");
    }

    /// Shutdown closes the door before it waits: nothing new is spawned or attached while the sessions
    /// it's waiting on persist, and it waits for all of them — including one the reaper had already
    /// stopped.
    #[tokio::test]
    async fn shutdown_refuses_new_attachments_and_waits_for_every_exit() {
        let probe = Arc::new(Probe::default());
        let sup = probe_supervisor(&probe, false);
        attach_and_detach(&sup, "reaped").await;
        let (idle_tx, idle) = {
            let p = sup.pin(Some("idle".into()), None).await.unwrap();
            (p.input_tx, p.id)
        };
        until("both sessions start", || probe.started() == 2).await;
        sup.reap_idle(Duration::ZERO);
        until("the reaped one is exiting", || probe.exiting() == 1).await;

        let shutdown = tokio::spawn({
            let sup = sup.clone();
            async move { sup.shutdown().await }
        });
        settle().await;
        // The still-attached connection's own sender keeps that session's input open; it closes on
        // teardown, as a real connection's does.
        drop(idle_tx);
        until("both are exiting", || probe.exiting() == 2).await;

        assert!(sup.pin(Some(idle), None).await.is_err());
        assert!(sup.pin(Some("fresh".into()), None).await.is_err());
        assert!(sup.pin(None, None).await.is_err());
        settle().await;
        assert_eq!(probe.started(), 2, "nothing spawned during shutdown");
        assert!(!shutdown.is_finished(), "shutdown waits for every exit");

        probe.release.cancel();
        shutdown.await.unwrap();
        assert!(lock_ignoring_poison(&sup.table).sessions.is_empty());
    }

    /// A session stopped while still `Starting` never runs its body — it never touches storage — and
    /// still frees its id.
    #[tokio::test]
    async fn a_session_stopped_before_it_goes_live_never_runs() {
        let probe = Arc::new(Probe::default());
        let sup = probe_supervisor(&probe, false);
        // No yield between spawn and reap: the task hasn't been polled, so it is still `Starting`.
        attach_and_detach(&sup, "s1").await;
        sup.reap_idle(Duration::ZERO);
        until("the id is freed", || {
            lock_ignoring_poison(&sup.table).sessions.is_empty()
        })
        .await;
        assert_eq!(probe.started(), 0);
    }

    /// Only a task's own incarnation is ever removed by its exit: a stale one leaves a newer entry
    /// alone, and still fires its own latch.
    #[test]
    fn an_exit_only_removes_its_own_incarnation() {
        let table: Arc<Mutex<Table>> = Arc::default();
        let (mut newer, _rx) = handle(1, None);
        newer.incarnation = 7;
        lock_ignoring_poison(&table)
            .sessions
            .insert("s1".into(), newer);

        let stale_exited = CancellationToken::new();
        drop(ExitGuard {
            table: table.clone(),
            id: "s1".into(),
            incarnation: 6,
            exited: stale_exited.clone(),
            metrics: None,
            end_reason: Arc::new(Mutex::new(crate::metrics::SessionEnd::Client)),
        });
        assert!(stale_exited.is_cancelled());
        assert!(lock_ignoring_poison(&table).sessions.contains_key("s1"));

        drop(ExitGuard {
            table: table.clone(),
            id: "s1".into(),
            incarnation: 7,
            exited: CancellationToken::new(),
            metrics: None,
            end_reason: Arc::new(Mutex::new(crate::metrics::SessionEnd::Client)),
        });
        assert!(!lock_ignoring_poison(&table).sessions.contains_key("s1"));
    }

    #[test]
    fn parse_http_head_reads_post_path_query_and_content_length() {
        let raw = b"POST /_beyond/agent?session_id=abc HTTP/1.1\r\nHost: localhost\r\nContent-Length: 12\r\n\r\n{\"type\":\"x\"}";
        let (head, n) = parse_http_head(raw).unwrap();
        assert_eq!(head.method, "POST");
        assert_eq!(head.path, "/_beyond/agent");
        assert_eq!(head.query.as_deref(), Some("session_id=abc"));
        assert_eq!(head.content_length, Some(12));
        assert_eq!(&raw[n..], b"{\"type\":\"x\"}");
        assert_eq!(
            session_id_from_query(head.query.as_deref()).as_deref(),
            Some("abc")
        );
    }

    #[test]
    fn parse_http_head_rejects_oversize_content_length_before_reading_the_body() {
        let raw = format!(
            "POST /_beyond/agent HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_INBOUND_MESSAGE_BYTES + 1
        );
        match parse_http_head(raw.as_bytes()) {
            Err(HttpError::PayloadTooLarge) => {}
            other => panic!("expected PayloadTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn parse_http_head_reports_incomplete_until_the_header_block_ends() {
        match parse_http_head(b"POST /_beyond/agent HTTP/1.1\r\nHost: localhost\r\n") {
            Err(HttpError::Incomplete) => {}
            other => panic!("expected Incomplete, got {other:?}"),
        }
    }

    #[test]
    fn reply_matches_requires_id_and_command() {
        let ack = json!({"type":"ack","command":"prompt","id":"p1"});
        let resp = json!({"type":"response","command":"prompt","id":"p1","success":true});
        let other = json!({"type":"response","command":"prompt","id":"p2","success":true});
        let event = json!({"type":"event","event":{"kind":"text"}});
        assert!(reply_matches(&ack, "prompt", "p1"));
        assert!(reply_matches(&resp, "prompt", "p1"));
        assert!(!reply_matches(&other, "prompt", "p1"));
        assert!(!reply_matches(&event, "prompt", "p1"));
        assert!(!reply_matches(&ack, "get_state", "p1"));
    }

    // ---- `/readyz`: one probe at a time, and a bounded wait ---------------------------------------

    /// The single-flight property. Before this, the memo was written only *after* a probe returned, so
    /// every request that arrived during one missed the cache and spawned another uncancellable
    /// `spawn_blocking` — which is how a hung mount plus a 10-second probe cadence drains tokio's
    /// 512-thread blocking pool and takes session listing down with it.
    #[tokio::test]
    async fn concurrent_readyz_probes_share_a_single_run() {
        let cache = ReadyCache::default();
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let answers = futures::future::join_all((0..8).map(|_| {
            let runs = runs.clone();
            cache.check_with(
                move || {
                    runs.fetch_add(1, Ordering::SeqCst);
                    // Long enough that every caller is waiting on the same probe, short enough that
                    // the test's own deadline is never the thing being measured.
                    std::thread::sleep(Duration::from_millis(150));
                    None
                },
                Duration::from_secs(30),
            )
        }))
        .await;

        assert_eq!(runs.load(Ordering::SeqCst), 1, "one probe, eight callers");
        assert!(answers.iter().all(Option::is_none), "{answers:?}");

        // And the successful answer is still memoized for the TTL, so the next caller probes nothing.
        assert_eq!(
            cache
                .check_with(
                    || panic!("a memoized answer must not probe"),
                    Duration::from_secs(30)
                )
                .await,
            None
        );
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    /// A probe that never answers must cost one blocking thread, not one per request: past the
    /// deadline the caller says "not ready" on its own and starts nothing.
    #[tokio::test]
    async fn a_readyz_probe_past_its_deadline_is_not_ready_and_starts_no_second_probe() {
        let cache = ReadyCache::default();
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let release = Arc::new(AtomicBool::new(false));

        for _ in 0..4 {
            let runs = runs.clone();
            let release = release.clone();
            let answer = cache
                .check_with(
                    move || {
                        runs.fetch_add(1, Ordering::SeqCst);
                        while !release.load(Ordering::SeqCst) {
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        None
                    },
                    Duration::from_millis(50),
                )
                .await;
            assert_eq!(answer.as_deref(), Some("shard probe timed out"));
        }
        assert_eq!(runs.load(Ordering::SeqCst), 1, "one hung probe, not four");

        // The timed-out probe's own answer still lands, so the replica recovers without a new one.
        release.store(true, Ordering::SeqCst);
        for _ in 0..1000 {
            if lock_ignoring_poison(&cache.0).inflight.is_none() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            cache
                .check_with(
                    || panic!("the finished probe's answer must be reused"),
                    Duration::from_millis(50)
                )
                .await,
            None
        );
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    // ---- The session lock: bounded, and no directory left behind ----------------------------------

    /// `spawn_blocking` cannot be cancelled, so a lock taken after the deadline has passed belongs to
    /// nobody. It must be released rather than held for a session no connection is waiting on.
    #[tokio::test]
    async fn a_session_lock_taken_past_the_deadline_is_released_not_leaked() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("s1.alpha");

        let slow = path.clone();
        let start = Instant::now();
        let timed_out = take_session_lock_within(Duration::from_millis(50), move || {
            std::thread::sleep(Duration::from_millis(300));
            std::fs::create_dir_all(&slow)?;
            acquire_session_lock(&slow)
        })
        .await;

        let Err(err) = timed_out else {
            panic!("the deadline must win")
        };
        assert_eq!(err.kind(), std::io::ErrorKind::TimedOut, "{err}");
        assert!(
            start.elapsed() < Duration::from_millis(250),
            "the caller must not wait out the blocking task: {:?}",
            start.elapsed()
        );

        // The orphan goes on to take the lock — and must then let it go. `acquire_session_lock`
        // reports a lock this process already holds as "held", so succeeding here is proof it did.
        for _ in 0..200 {
            if acquire_session_lock(&path).ok().flatten().is_some() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("the orphaned lock was never released");
    }

    /// A start that never wrote a segment gives back the directory `take_session_lock` had to create
    /// in order to hold the lock at all — otherwise every refused or failed start leaves an empty
    /// `<id>/` plus its `lock` on the shard forever.
    #[tokio::test]
    async fn an_empty_session_directory_is_taken_back_with_its_lock() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("s1.alpha");

        let lock = take_session_lock(path.clone()).await.unwrap();
        assert!(lock.is_some() && path.is_dir());
        release_session_lock(lock, Some(path.clone())).await;
        assert!(
            !path.exists(),
            "an empty session directory must not outlive its start"
        );
    }

    /// The other half of the same rule: a directory that holds a segment is a real session, so it is
    /// left exactly as it is — `remove_dir` on a non-empty directory is the atomic check that says so.
    #[tokio::test]
    async fn a_session_directory_with_a_segment_is_left_alone() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("s1.alpha");

        let lock = take_session_lock(path.clone()).await.unwrap();
        assert!(lock.is_some());
        std::fs::write(path.join("000001.jsonl"), b"{}\n").unwrap();
        release_session_lock(lock, Some(path.clone())).await;

        assert!(
            path.join("000001.jsonl").is_file(),
            "the segment must survive"
        );
        assert!(
            path.join("lock").is_file(),
            "so must the lock file it is locked through"
        );
        // The lock itself is released, so the next owner can take it.
        assert!(acquire_session_lock(&path).unwrap().is_some());
    }
}
