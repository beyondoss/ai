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
//! supervisor's routing key **and** the persisted session id: it's handed to the session as
//! [`ServeConfig::session_id`], which *addresses* it in the repo — open that session, or create it under
//! exactly that id ([`crate::session_store::SessionRepo::open_or_create_id`]). So the id is stable
//! across reconnects, and a cold reconnect after a full process restart reopens the same conversation
//! rather than a blank one. `--no-session-persistence` opts out into in-memory-only sessions, which
//! still live re-attach for the process's lifetime.
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
//! There is none here, by design: the agent authenticates no caller. Bind **loopback/internal only**
//! and trust the front door (the edge, in another repo) to have validated the client before forwarding
//! the upgrade. This module never sees or parses a user token.

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures::{SinkExt, StreamExt};
use serde_json::json;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, TcpStream};
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;
use tokio::time::{Duration, MissedTickBehavior};
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::handshake::server::{ErrorResponse, Request, Response};
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::{Message, Utf8Bytes};
use tokio_util::sync::CancellationToken;

use crate::serve::{
    OutFanout, OutFrame, OutSink, ServeConfig, SharedOutConn, Signal, UpstreamHttp2, frame_to_line,
    lock_ignoring_poison, serve_session,
};
use crate::session_store::{is_valid_session_id, new_id, scan_listings, scan_session_dir};

/// The fixed URL path a WebSocket upgrade must target. The front door maps a service subdomain to this
/// path on the loopback listener; any other path is rejected at the handshake.
const WS_PATH: &str = "/_beyond/agent";

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

/// How long a **detached** session (no attached connection) stays live before the reaper reclaims it,
/// when the operator gave no `--session-idle-timeout`. The default has to be finite: without a reaper
/// the session map only grows — a connection that omits `?session_id=` mints a fresh id, and every id
/// owns an OS thread, an `Agent`, and a gateway pool until the daemon stops. It also has to be *long*,
/// because a detached session is not a dead one: re-attaching to a still-running run is the entire point
/// of the design (see the module doc), so the window must comfortably outlast a tunnel, a locked screen,
/// or a lunch break. An hour is both. Nothing is lost when it fires — a reaped session persisted on its
/// way out, and reconnecting to its id respawns it and replays from disk.
const DEFAULT_SESSION_IDLE_TIMEOUT: Duration = Duration::from_secs(60 * 60);

/// How long a batch of session threads gets to persist and exit before the caller stops waiting on them.
const JOIN_GRACE: Duration = Duration::from_secs(10);

/// How often [`join_handles_within`] re-checks whether the session threads it's waiting on have exited.
/// Short enough not to add perceptible latency to a graceful shutdown (the common case: every session
/// persists in milliseconds), long enough to cost nothing while waiting.
const JOIN_POLL: Duration = Duration::from_millis(10);

/// A live session, reachable by id across connections. The retained `input_tx` is what makes a
/// dropped socket *not* an EOF — the session's `input_rx.recv()` pends until the next command instead
/// of shutting down (see the module doc).
struct SessionHandle {
    /// Feeds command lines into the session's [`serve_session`] loop. Held here (not by any socket) so
    /// the session outlives its connections. Every attached connection feeds this one channel, so any
    /// device can drive the session (its `mpsc` is multi-sender).
    input_tx: mpsc::Sender<String>,
    /// The session's set of attached connections — its output is **broadcast** to all of them, so a
    /// phone and a TUI (or any N of the user's devices) on one session all see the live stream at once
    /// (see [`crate::serve::OutFanout`]). Each connection registers its sink on attach and removes it on
    /// disconnect.
    out_conn: SharedOutConn,
    /// The session's dedicated thread. Retained so a graceful shutdown can **wait** for the session to
    /// persist and exit (dropping `input_tx` closes its input, then this joins) rather than letting
    /// `process::exit` race the persist. `None` only transiently while a handle is being moved out.
    join: Option<std::thread::JoinHandle<()>>,
    /// How many connections are currently attached. The idle reaper only considers a session for
    /// reclamation when this reaches `0` (see [`Self::last_detached_at`]).
    attached: usize,
    /// When the last connection detached (`attached` reached `0`), for the idle reaper's clock. `None`
    /// while any connection is attached, or on a fresh session — either way it isn't reap-eligible yet.
    last_detached_at: Option<Instant>,
    /// `true` exactly while the session's [`serve_session`] loop is running a `prompt`. The reaper reads
    /// it so a detached-but-mid-run background session is never reaped out from under an in-flight turn.
    running: Arc<AtomicBool>,
}

/// Owns the `session id → live session` map and the base config every session is cloned from.
struct Supervisor {
    sessions: Mutex<HashMap<String, SessionHandle>>,
    cfg: ServeConfig,
}

impl Supervisor {
    /// Derive a per-session config: address the session by its routing key and drop `listen`.
    ///
    /// Pinning `session_id` is the whole mechanism — repo mode opens exactly that session or creates it
    /// under exactly that id, so the persisted id always equals the routing key. This used to rewrite
    /// each session into single-file mode at `<session-dir>/<id>.jsonl` instead, purely to dodge repo
    /// mode's old behavior of resolving by `cwd` and collapsing every session in a directory onto one.
    /// With an id now taking precedence over the cwd match that workaround is unnecessary, and dropping
    /// it fixes what it cost: daemon files were named `<id>.jsonl` where the repo names its own
    /// `<created_at>_<id>.jsonl`, so `find_path`'s `_<id>.jsonl` lookup couldn't see them — a daemon
    /// session appeared in `list_sessions` but `switch_session` reported it missing.
    fn session_cfg(&self, id: &str) -> ServeConfig {
        let mut c = self.cfg.clone();
        c.listen = None;
        // A spawned session must never itself re-bind a transport listener — it's driven purely
        // through its `input_rx`/`out_conn` channels by the supervisor.
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

    /// Attach `ws` to the session named `requested_id` (minting a fresh id if `None`), spawning the
    /// session if it isn't already live. Supersedes any previous connection to that session
    /// (last-attach-wins), then drives this socket until it closes or is superseded — the session
    /// itself keeps running either way.
    async fn attach<S>(self: &Arc<Self>, requested_id: Option<String>, ws: WebSocketStream<S>)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let id = requested_id.unwrap_or_else(new_id);

        // Look up (or spawn) the session and register this connection — all under the map lock, which is
        // never held across an `.await`. No eviction: multiple connections coexist on one session (the
        // output fans out to all; input is shared), so a phone and a TUI can watch/drive it together.
        let (input_tx, out_conn) = {
            let mut sessions = lock_ignoring_poison(&self.sessions);

            // A handle whose session task has ended (SIGTERM, internal error) has a closed `input_tx`;
            // treat it as absent and respawn so a reconnect to that id still works.
            if sessions.get(&id).is_some_and(|h| h.input_tx.is_closed()) {
                sessions.remove(&id);
            }

            let handle = sessions.entry(id.clone()).or_insert_with(|| {
                let (input_tx, input_rx) = mpsc::channel::<String>(crate::serve::IN_CHANNEL_BOUND);
                let out_conn: SharedOutConn = Arc::new(Mutex::new(OutFanout::default()));
                let cfg = self.session_cfg(&id);
                let session_out = out_conn.clone();
                let log_id = id.clone();
                // Shared with the session loop: `true` only while it's running a `prompt`. The reaper
                // reads this handle-side clone to never reclaim a mid-run background session.
                let running = Arc::new(AtomicBool::new(false));
                let session_running = running.clone();
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
                        match serve_session(cfg, input_rx, session_out, session_running).await {
                            Ok(_) => {}
                            Err(e) => eprintln!("serve: session {log_id} ended: {e}"),
                        }
                    });
                });
                SessionHandle {
                    input_tx,
                    out_conn,
                    join: Some(join),
                    attached: 0,
                    last_detached_at: None,
                    running,
                }
            });

            // Register, don't evict: one more attached connection, and clear any detach timestamp so the
            // reaper's clock only runs while genuinely detached (`attached == 0`).
            handle.attached += 1;
            handle.last_detached_at = None;
            (handle.input_tx.clone(), handle.out_conn.clone())
        };

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
                                        let frame = self.list_daemon_sessions(client_id).await;
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
        {
            let mut sessions = lock_ignoring_poison(&self.sessions);
            if let Some(h) = sessions.get_mut(&id) {
                h.attached = h.attached.saturating_sub(1);
                if h.attached == 0 {
                    h.last_detached_at = Some(Instant::now());
                }
            }
        }
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
        join_handles(joins).await;
    }

    /// Answer a `list_daemon_sessions` command: the union of every session the daemon knows about — the
    /// live in-memory map (`live:true`) and every `*.jsonl` under the base `--session-dir` (whose
    /// `live` flag says whether that persisted id also has a running task right now). The reply is a
    /// single `response` frame the caller sends back on the originating connection.
    async fn list_daemon_sessions(&self, client_id: Option<String>) -> OutFrame {
        // Snapshot the live ids (task still running ⇒ `input_tx` open) under the lock, then drop it —
        // the on-disk scan below must not run while holding the map mutex.
        let live: HashSet<String> = {
            let sessions = lock_ignoring_poison(&self.sessions);
            sessions
                .iter()
                .filter(|(_, h)| !h.input_tx.is_closed())
                .map(|(id, _)| id.clone())
                .collect()
        };

        // On-disk listings, if this daemon persists at all. `scan_listings` is CPU-bound and uses
        // `thread::scope`, so it runs on the blocking pool rather than stalling this async task.
        let metas = match &self.cfg.session_dir {
            Some(dir) => {
                let paths = scan_session_dir(std::path::Path::new(dir));
                tokio::task::spawn_blocking(move || scan_listings(paths, &|_, _| {}))
                    .await
                    .unwrap_or_default()
            }
            None => Vec::new(),
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

    /// Idle reaper: reclaim every session [`is_reapable`] names. Removing the handle drops its retained
    /// `input_tx`, so the session observes EOF and persists+exits exactly as in [`shutdown`]; the join
    /// then waits for that persist to land (and reclaims the thread of one that had already exited). A
    /// reconnect to a just-reaped id transparently respawns and replays via `get_messages{since}`.
    async fn reap_idle(&self, timeout: Duration) {
        let joins: Vec<std::thread::JoinHandle<()>> = {
            let mut sessions = lock_ignoring_poison(&self.sessions);
            let reap: Vec<String> = sessions
                .iter()
                .filter(|(_, h)| is_reapable(h, timeout))
                .map(|(id, _)| id.clone())
                .collect();
            reap.into_iter()
                .filter_map(|id| sessions.remove(&id).and_then(|mut h| h.join.take()))
                .collect()
        };
        join_handles(joins).await;
    }
}

/// Whether the reaper should reclaim this session.
///
/// A **closed `input_tx`** is the strongest reason of all: the session's loop is already gone (it
/// returned early — no credential, an unwritable session dir — or hit an internal error), so nothing
/// will ever read its input again. The entry is pure garbage: a `HashMap` slot, a `SharedOutConn`, and
/// an unreclaimed `JoinHandle` for a thread that has already exited. Reap it whatever its attach state
/// or clock says — a live connection still pinned to it is no reason to keep a corpse (that connection's
/// next command fails its `input_tx.send` and tears the socket down; a reconnect respawns the id).
///
/// Otherwise the session is alive, and the ordinary conditions apply: **detached** (`attached == 0`) for
/// at least `timeout`, and not mid-`prompt` — a detached background run is exactly what this design
/// exists to keep alive, so `running` is never reaped out from under an in-flight turn.
fn is_reapable(h: &SessionHandle, timeout: Duration) -> bool {
    if h.input_tx.is_closed() {
        return true;
    }
    h.attached == 0
        && h.last_detached_at.is_some_and(|d| d.elapsed() >= timeout)
        && !h.running.load(Ordering::Relaxed)
}

/// Wait for a batch of session threads to finish persisting, bounded by [`JOIN_GRACE`] so a wedged
/// session can't hang the caller forever. Shared by [`Supervisor::shutdown`] and
/// [`Supervisor::reap_idle`] so both persist-then-join on the same discipline.
async fn join_handles(joins: Vec<std::thread::JoinHandle<()>>) {
    join_handles_within(joins, JOIN_GRACE).await;
}

/// The waiting itself, with the grace period as a parameter so it can be tested.
///
/// The wait **polls `is_finished`** rather than parking a blocking join behind a timeout, because a
/// timeout only abandons the *await* — the blocking join underneath it keeps running. Wedge one session
/// and a `spawn_blocking(|| handle.join())` would sit on one of tokio's blocking threads (a finite pool)
/// for the rest of the process's life; wedge enough of them and every `spawn_blocking` in the daemon
/// stalls behind an empty pool, session persistence included. `join()` is therefore only ever called on
/// a thread that has *already exited*, where it cannot block: it just reaps the thread.
///
/// A straggler past `grace` is dropped, which detaches it — the OS reclaims its stack when it finally
/// does exit, and nothing (no caller, no pool thread) is left waiting on it. That is all the caller can
/// do: a graceful shutdown falls through to `process::exit` regardless, and the reaper has already
/// removed the id from the map, so it will never see that session again.
async fn join_handles_within(joins: Vec<std::thread::JoinHandle<()>>, grace: Duration) {
    if joins.is_empty() {
        return;
    }
    let deadline = Instant::now() + grace;
    let mut pending = joins;
    loop {
        let mut still = Vec::with_capacity(pending.len());
        for j in pending {
            if j.is_finished() {
                let _ = j.join();
            } else {
                still.push(j);
            }
        }
        pending = still;
        if pending.is_empty() {
            return;
        }
        if Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(JOIN_POLL).await;
    }
    eprintln!(
        "serve: {} session(s) did not persist within the join grace period",
        pending.len()
    );
}

/// Resolve the idle reaper's window from [`ServeConfig::session_idle_timeout`]: unset ⇒
/// [`DEFAULT_SESSION_IDLE_TIMEOUT`] (the reaper is *on* by default — the map has no other way to shrink,
/// see the const), and `0` ⇒ `None`, the explicit opt-out for an operator who genuinely wants every
/// session pinned for the daemon's lifetime.
fn resolve_idle_timeout(configured: Option<Duration>) -> Option<Duration> {
    match configured {
        Some(t) if t.is_zero() => None,
        Some(t) => Some(t),
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
    let idle_timeout = resolve_idle_timeout(cfg.session_idle_timeout);

    let supervisor = Arc::new(Supervisor {
        sessions: Mutex::new(HashMap::new()),
        cfg,
    });
    let mut shutdown = crate::serve::ShutdownSignal::new()?;

    // The idle reaper (on unless `--session-idle-timeout 0` turned it off). A background ticker that
    // reclaims dead and detached-idle-not-mid-run sessions — the same drop-`input_tx` → persist → join
    // path shutdown uses. Its handle is aborted on shutdown so the process can exit cleanly. Tick at
    // half the timeout (so a session is reaped within ~1.5× the timeout at worst), capped at 30s so a
    // long timeout still ticks at a sane cadence.
    if idle_timeout.is_none() {
        eprintln!(
            "serve: idle-session reaper OFF (--session-idle-timeout 0) — every session, including one \
             no client ever re-attaches to, holds its thread and gateway pool until the daemon stops"
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
                supervisor.reap_idle(t).await;
            }
        })
    });

    loop {
        tokio::select! {
            sig = shutdown.wait() => {
                eprintln!("serve: shutting down websocket listener");
                // Stop the idle reaper before draining sessions ourselves, so the two don't race over
                // the same handles.
                if let Some(reaper) = &reaper {
                    reaper.abort();
                }
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
    // Every inbound message becomes one queued command line, so cap what a single one can cost us —
    // tungstenite's defaults (64 MiB message / 16 MiB frame) let a client hand the session queue tens of
    // MB at a time. Oversize is a hard protocol error (the connection closes); nothing is truncated.
    let config = WebSocketConfig::default()
        .max_message_size(Some(MAX_INBOUND_MESSAGE_BYTES))
        .max_frame_size(Some(MAX_INBOUND_MESSAGE_BYTES));
    let ws = tokio_tungstenite::accept_hdr_async_with_config(
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
        Some(config),
    )
    .await?;

    let requested_id = lock_ignoring_poison(&requested_id).take();
    // A client-supplied id becomes a filename component (`<id>.jsonl`) — reject anything that isn't a
    // safe session id rather than letting it escape the sessions directory.
    if let Some(id) = &requested_id
        && !is_valid_session_id(id)
    {
        let mut ws = ws;
        let _ = ws.close(None).await;
        return Err(format!("invalid session_id: {id:?}").into());
    }

    supervisor.attach(requested_id, ws).await;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// Build a handle with no session behind it. `alive` keeps the input channel open (the receiver is
    /// returned and must be held); dropping the receiver is how a test models a session whose loop ended.
    fn handle(
        attached: usize,
        detached_ago: Option<Duration>,
    ) -> (SessionHandle, mpsc::Receiver<String>) {
        let (input_tx, input_rx) = mpsc::channel::<String>(crate::serve::IN_CHANNEL_BOUND);
        let h = SessionHandle {
            input_tx,
            out_conn: Arc::new(Mutex::new(OutFanout::default())),
            join: None,
            attached,
            last_detached_at: detached_ago.and_then(|d| Instant::now().checked_sub(d)),
            running: Arc::new(AtomicBool::new(false)),
        };
        (h, input_rx)
    }

    #[test]
    fn idle_timeout_defaults_to_a_finite_window_and_zero_opts_out() {
        assert_eq!(
            resolve_idle_timeout(None),
            Some(DEFAULT_SESSION_IDLE_TIMEOUT),
            "no --session-idle-timeout must still reap: the map has no other way to shrink"
        );
        assert_eq!(
            resolve_idle_timeout(Some(Duration::ZERO)),
            None,
            "0 opts out"
        );
        assert_eq!(
            resolve_idle_timeout(Some(Duration::from_secs(5))),
            Some(Duration::from_secs(5))
        );
    }

    #[test]
    fn a_dead_session_is_reapable_however_it_looks_otherwise() {
        // Its loop ended (input receiver gone) while a connection is *still attached* and its idle clock
        // never started: every ordinary condition says "keep", and it must still be reaped — otherwise
        // the entry, its fanout, and its exited thread's handle are retained for the daemon's life.
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
    }

    /// A wedged session must not hold the caller (nor a blocking-pool thread) past the grace period, and
    /// the threads that *did* exit must still be reaped in the same pass.
    #[tokio::test]
    async fn a_wedged_thread_does_not_hold_the_join_past_the_grace_period() {
        let release = Arc::new(AtomicBool::new(false));
        let wedged = {
            let release = release.clone();
            std::thread::spawn(move || {
                while !release.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(5));
                }
            })
        };
        let finished = std::thread::spawn(|| {});

        let start = Instant::now();
        join_handles_within(vec![finished, wedged], Duration::from_millis(200)).await;
        let waited = start.elapsed();

        assert!(
            waited >= Duration::from_millis(200) && waited < Duration::from_secs(5),
            "the wait must end at the grace period, not when the wedged session finally exits: {waited:?}"
        );
        release.store(true, Ordering::Relaxed);
    }

    #[tokio::test]
    async fn joining_returns_as_soon_as_every_thread_has_exited() {
        let threads: Vec<_> = (0..4).map(|_| std::thread::spawn(|| {})).collect();
        let start = Instant::now();
        join_handles_within(threads, Duration::from_secs(10)).await;
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "finished threads must not wait out the grace period"
        );
    }
}
