//! `serve --listen <addr>` — the [`serve`](crate::serve) control protocol offered over a WebSocket
//! instead of stdio, for a client (the Beyond iPhone app) that speaks one JSON object per WS **text**
//! message. The protocol is **byte-identical** to stdio mode; this module is a thin transport adapter
//! over the same [`serve_session`](crate::serve::serve_session) core.
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
//! is minted; the client learns it from any `response`/`get_state` frame). That id is both the
//! supervisor's routing key **and** the persisted session id: each WS session is its own JSONL file
//! `<session-dir>/<id>.jsonl` (via [`ServeConfig::session_id`]/`session_file`), so the id is stable
//! across reconnects and a cold reconnect after a process restart re-opens the same file. Without a
//! `--session-dir`, sessions are in-memory only (live re-attach still works for the process's
//! lifetime). Repo-mode multi-session commands (`list_sessions`, `switch_session`) are not used — each
//! connection is pinned to one session by its URL.
//!
//! ## Auth
//!
//! There is none here, by design: the agent authenticates no caller. Bind **loopback/internal only**
//! and trust the front door (the edge, in another repo) to have validated the client before forwarding
//! the upgrade. This module never sees or parses a user token.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use futures::{SinkExt, StreamExt};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, TcpStream};
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;
use tokio::time::{Duration, MissedTickBehavior};
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::handshake::server::{ErrorResponse, Request, Response};
use tokio_util::sync::CancellationToken;

use crate::serve::{
    OutFrame, ServeConfig, SharedOutConn, Signal, frame_to_line, lock_ignoring_poison,
    serve_session,
};
use crate::session_store::{is_valid_session_id, new_id};

/// The fixed URL path a WebSocket upgrade must target. The front door maps a service subdomain to this
/// path on the loopback listener; any other path is rejected at the handshake.
const WS_PATH: &str = "/_beyond/agent";

/// How often the server sends an unsolicited `Ping` so an idle mobile connection isn't reaped by
/// NAT/proxies. Also the granularity at which a wholly-dead socket is noticed (the ping send fails).
const PING_INTERVAL: Duration = Duration::from_secs(30);

/// A live session, reachable by id across connections. The retained `input_tx` is what makes a
/// dropped socket *not* an EOF — the session's `input_rx.recv()` pends until the next command instead
/// of shutting down (see the module doc).
struct SessionHandle {
    /// Feeds command lines into the session's [`serve_session`] loop. Held here (not by any socket) so
    /// the session outlives its connections.
    input_tx: mpsc::UnboundedSender<String>,
    /// The session's swappable output tail: the supervisor rebinds this to each newly-attached
    /// connection's send channel, or leaves the previous one (whose receiver is gone) to drop frames.
    out_conn: SharedOutConn,
    /// The currently-attached connection's cancellation token, if any. A new attach cancels it
    /// (last-attach-wins) so a stale socket from a flaky reconnect can't keep feeding the session.
    conn_cancel: Option<CancellationToken>,
    /// The session's dedicated thread. Retained so a graceful shutdown can **wait** for the session to
    /// persist and exit (dropping `input_tx` closes its input, then this joins) rather than letting
    /// `process::exit` race the persist. `None` only transiently while a handle is being moved out.
    join: Option<std::thread::JoinHandle<()>>,
}

/// Owns the `session id → live session` map and the base config every session is cloned from.
struct Supervisor {
    sessions: Mutex<HashMap<String, SessionHandle>>,
    cfg: ServeConfig,
}

impl Supervisor {
    /// Derive a per-session config: pin the id, drop `listen`, and give the session its own JSONL file
    /// under the base `--session-dir` so its persisted id equals its routing key. No base dir ⇒
    /// in-memory only (live re-attach still works while the task lives).
    fn session_cfg(&self, id: &str) -> ServeConfig {
        let mut c = self.cfg.clone();
        c.listen = None;
        // A spawned session must never itself re-bind a transport listener — it's driven purely
        // through its `input_rx`/`out_conn` channels by the supervisor.
        c.listen_uds = None;
        c.listen_uds_mode = None;
        c.session_id = Some(id.to_string());
        match &self.cfg.session_dir {
            Some(dir) => {
                c.session_file = Some(
                    std::path::Path::new(dir)
                        .join(format!("{id}.jsonl"))
                        .to_string_lossy()
                        .into_owned(),
                );
                c.session_dir = None;
            }
            None => {
                c.no_session_persistence = true;
                c.session_file = None;
            }
        }
        c
    }

    /// Attach `ws` to the session named `requested_id` (minting a fresh id if `None`), spawning the
    /// session if it isn't already live. Supersedes any previous connection to that session
    /// (last-attach-wins), then drives this socket until it closes or is superseded — the session
    /// itself keeps running either way.
    async fn attach<S>(self: &Arc<Self>, requested_id: Option<String>, ws: WebSocketStream<S>)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let id = requested_id.unwrap_or_else(new_id);

        // Look up (or spawn) the session and supersede any prior connection — all under the map lock,
        // which is never held across an `.await`.
        let (input_tx, out_conn, my_cancel) = {
            let mut sessions = lock_ignoring_poison(&self.sessions);

            // A handle whose session task has ended (SIGTERM, internal error) has a closed `input_tx`;
            // treat it as absent and respawn so a reconnect to that id still works.
            if sessions.get(&id).is_some_and(|h| h.input_tx.is_closed()) {
                sessions.remove(&id);
            }

            let handle = sessions.entry(id.clone()).or_insert_with(|| {
                let (input_tx, input_rx) = mpsc::unbounded_channel::<String>();
                let out_conn: SharedOutConn = Arc::new(Mutex::new(None));
                let cfg = self.session_cfg(&id);
                let session_out = out_conn.clone();
                let log_id = id.clone();
                // `serve_session`'s event sink is a `Box<dyn FnMut>` (not `Send`), so its future
                // can't be `tokio::spawn`ed onto the multi-threaded accept runtime — the stdio path
                // only ever `.await`s it inline. Give each session its own thread with a
                // current-thread runtime instead; the `mpsc` channels bridging it to the accept
                // runtime are runtime-agnostic. Sessions are few (one per connected client), so a
                // thread apiece is fine.
                let join = std::thread::spawn(move || {
                    let rt = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(rt) => rt,
                        Err(e) => {
                            eprintln!("serve: session {log_id} runtime build failed: {e}");
                            return;
                        }
                    };
                    rt.block_on(async move {
                        match serve_session(cfg, input_rx, session_out).await {
                            Ok(_) => {}
                            Err(e) => eprintln!("serve: session {log_id} ended: {e}"),
                        }
                    });
                });
                SessionHandle {
                    input_tx,
                    out_conn,
                    conn_cancel: None,
                    join: Some(join),
                }
            });

            // Last-attach-wins: cancel whoever was attached before us.
            if let Some(prev) = handle.conn_cancel.take() {
                prev.cancel();
            }
            let my_cancel = CancellationToken::new();
            handle.conn_cancel = Some(my_cancel.clone());
            (handle.input_tx.clone(), handle.out_conn.clone(), my_cancel)
        };

        // Bind this connection's send channel as the session's output tail. Rebinding drops the prior
        // `conn_tx`, which closes the prior send task's receiver (a second teardown path alongside the
        // cancel token above).
        let (conn_tx, mut conn_rx) = mpsc::unbounded_channel::<OutFrame>();
        *lock_ignoring_poison(&out_conn) = Some(conn_tx);

        let (mut sink, mut stream) = ws.split();

        // The send task owns the socket's write half. Everything outbound — protocol frames, `Pong`
        // replies, and periodic keepalive `Ping`s — funnels through it so the single sink has one
        // writer. `ctrl_rx` carries the read loop's `Pong` replies.
        let (ctrl_tx, mut ctrl_rx) = mpsc::unbounded_channel::<Message>();
        let send_cancel = my_cancel.clone();
        let send_task = tokio::spawn(async move {
            let mut ping = tokio::time::interval(PING_INTERVAL);
            ping.set_missed_tick_behavior(MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    biased;
                    _ = send_cancel.cancelled() => break,
                    // A closed `conn_rx` means we were superseded (out_conn rebound) or the session
                    // ended — stop writing to this socket.
                    frame = conn_rx.recv() => match frame {
                        Some(frame) => if let Some(line) = frame_to_line(frame) {
                            if sink.send(Message::Text(line.into())).await.is_err() {
                                break;
                            }
                        },
                        None => break,
                    },
                    ctrl = ctrl_rx.recv() => {
                        if let Some(msg) = ctrl {
                            if sink.send(msg).await.is_err() {
                                break;
                            }
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
        // into the session; the session (not this loop) decides what to do with it.
        loop {
            tokio::select! {
                biased;
                _ = my_cancel.cancelled() => break,
                msg = stream.next() => {
                    let Some(msg) = msg else { break }; // socket closed
                    let msg = match msg {
                        Ok(m) => m,
                        Err(_) => break,
                    };
                    match msg {
                        Message::Text(text) => {
                            if input_tx.send(text.as_str().to_owned()).is_err() {
                                break; // session gone
                            }
                        }
                        Message::Ping(payload) => {
                            let _ = ctrl_tx.send(Message::Pong(payload));
                        }
                        Message::Close(_) => break,
                        // Pong (keepalive ack), Binary (protocol is text JSON), and any future frame
                        // kind are ignored.
                        _ => {}
                    }
                }
            }
        }

        // This connection is done; stop its send task. The session keeps running — a later connection
        // re-attaches by id. We deliberately do NOT reset `out_conn` to `None`: the next attach
        // overwrites it, and until then the session's writer harmlessly drops frames into this now-dead
        // channel.
        my_cancel.cancel();
        let _ = send_task.await;
    }

    /// Graceful shutdown: drain the session map (dropping each `input_tx`, which closes that session's
    /// input channel → it cancels any in-flight run, persists, and exits), then wait for every session
    /// thread to finish so persistence is durable before the process exits. Bounded so a wedged session
    /// can't hang the shutdown forever — a straggler is left to `process::exit`.
    async fn shutdown(&self) {
        let joins: Vec<std::thread::JoinHandle<()>> = {
            let mut sessions = lock_ignoring_poison(&self.sessions);
            sessions
                .drain()
                .filter_map(|(_, mut h)| h.join.take())
                .collect()
        };
        if joins.is_empty() {
            return;
        }
        // `JoinHandle::join` blocks, so hand it to a blocking thread and cap the total wait.
        let waiter = tokio::task::spawn_blocking(move || {
            for j in joins {
                let _ = j.join();
            }
        });
        if tokio::time::timeout(Duration::from_secs(10), waiter)
            .await
            .is_err()
        {
            eprintln!("serve: some sessions did not persist within the shutdown grace period");
        }
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

/// Serve the control protocol over the WebSocket listeners in `listeners` (loopback TCP and/or a
/// Unix-domain socket), all fronting **one** shared supervisor so a session is reachable over either
/// transport by its id. Runs until an OS shutdown signal, returning it so the caller exits with the
/// matching code — same contract as [`serve`](crate::serve::serve). Each accepted connection is
/// handled concurrently and routed to its session by id.
pub async fn serve_ws(
    cfg: ServeConfig,
    listeners: ServeListeners,
) -> Result<Option<Signal>, Box<dyn std::error::Error>> {
    let tcp_listener = match listeners.tcp {
        Some(addr) => {
            let listener = TcpListener::bind(addr).await?;
            let local = listener.local_addr()?;
            // A well-defined line (on stderr, which the protocol never uses) so an operator — or a
            // test that binds port 0 — can learn the actual bound address.
            eprintln!("serve: websocket listening on {local} (path {WS_PATH})");
            Some(listener)
        }
        None => None,
    };

    #[cfg(unix)]
    let uds_path = listeners.uds.clone();
    #[cfg(unix)]
    let uds_listener = match &listeners.uds {
        Some(path) => {
            let listener = bind_uds(path, listeners.uds_mode).await?;
            eprintln!(
                "serve: unix socket listening on {} (path {WS_PATH})",
                path.display()
            );
            Some(listener)
        }
        None => None,
    };
    // On a non-unix target there is no UDS to bind (`--listen-uds` errored before we got here); the
    // `select!` UDS arm still needs a listener binding, so give it a `None` that `pending()`s forever.
    #[cfg(not(unix))]
    let uds_listener: Option<()> = None;

    let supervisor = Arc::new(Supervisor {
        sessions: Mutex::new(HashMap::new()),
        cfg,
    });
    let mut shutdown = crate::serve::ShutdownSignal::new()?;

    loop {
        tokio::select! {
            sig = shutdown.wait() => {
                eprintln!("serve: shutting down websocket listener");
                // Best-effort remove the socket file so a restart isn't tripped by our own stale node.
                #[cfg(unix)]
                if let Some(path) = &uds_path {
                    let _ = std::fs::remove_file(path);
                }
                // Drive shutdown deterministically from here rather than relying on each session's own
                // signal handler (which runs on a spawned-thread runtime that may not receive the
                // signal): drop every retained `input_tx` so each session observes EOF and
                // cancels+persists+exits, then join its thread so persistence actually completes before
                // the caller's `process::exit`.
                supervisor.shutdown().await;
                return Ok(Some(sig));
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

/// Perform the WebSocket handshake (validating the path and extracting `?session_id=`), then hand the
/// connection to the supervisor to attach to its session. Generic over the underlying byte stream so
/// the same handshake+attach path serves both TCP and Unix-domain sockets.
async fn handle_connection<S>(
    supervisor: &Arc<Supervisor>,
    stream: S,
) -> Result<(), Box<dyn std::error::Error>>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    // The handshake callback sees the HTTP request. Reject a wrong path outright; stash the requested
    // session id (parsed from the query) for use after the upgrade completes.
    let requested_id: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let slot = requested_id.clone();
    let ws = tokio_tungstenite::accept_hdr_async(
        stream,
        move |req: &Request, resp: Response| -> Result<Response, ErrorResponse> {
            let uri = req.uri();
            if uri.path() != WS_PATH {
                // Build the rejection without the fallible `ResponseBuilder` (`http::Response::new` is
                // infallible, avoiding a bare `expect` on a value that can't fail).
                let mut err: ErrorResponse =
                    http::Response::new(Some(format!("not found: expected {WS_PATH}")));
                *err.status_mut() = http::StatusCode::NOT_FOUND;
                return Err(err);
            }
            if let Some(query) = uri.query() {
                for (k, v) in url::form_urlencoded::parse(query.as_bytes()) {
                    if k == "session_id" {
                        *lock_ignoring_poison(&slot) = Some(v.into_owned());
                    }
                }
            }
            Ok(resp)
        },
    )
    .await?;

    let requested_id = lock_ignoring_poison(&requested_id).take();
    // A client-supplied id becomes a filename component (`<id>.jsonl`) — reject anything that isn't a
    // safe session id rather than letting it escape the sessions directory.
    if let Some(id) = &requested_id {
        if !is_valid_session_id(id) {
            let mut ws = ws;
            let _ = ws.close(None).await;
            return Err(format!("invalid session_id: {id:?}").into());
        }
    }

    supervisor.attach(requested_id, ws).await;
    Ok(())
}
