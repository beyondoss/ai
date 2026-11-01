//! The writer side of a session: `OutFrame`, the redacting `OutTx`, and `OutFanout`, which broadcasts frames to
//! the attached connections and keeps what a re-attaching connection is caught up with.

use crate::serve::*;

/// Strip replica-host paths out of a failed `response` frame's `error` text.
///
/// Only service mode redacts, and only a `response` with `success: false` and a string `error` is
/// even inspected — an `OutFrame::Raw` (every streamed event, the hot path) short-circuits on its
/// variant, and a successful response on one map lookup. Error text is where host paths actually
/// leak: `std::io::Error` from the session store names the file it failed on.
pub(super) fn redact_frame(mode: &Mode, frame: OutFrame) -> OutFrame {
    if !mode.redacts_host_paths() {
        return frame;
    }
    let OutFrame::Value(mut v) = frame else {
        return frame;
    };
    if let Some(map) = v.as_object_mut()
        && map.get("success") == Some(&Value::Bool(false))
        && let Some(Value::String(error)) = map.get("error")
    {
        let redacted = mode.redact(error);
        map.insert("error".into(), Value::String(redacted));
    }
    OutFrame::Value(v)
}

/// The session's one way to the writer. `send` runs [`redact_frame`] for the session's mode, and
/// the raw channel sender never leaves this type, so no loop, spawned task or forwarder can reach
/// the tenant with an unredacted failed response. Clones carry the mode along.
#[derive(Clone)]
pub(crate) struct OutTx {
    pub(super) tx: mpsc::UnboundedSender<OutFrame>,
    pub(super) mode: Mode,
}

/// An [`OutTx`] that does not keep the writer alive (see [`OutTx::downgrade`]).
pub(super) struct WeakOutTx {
    pub(super) tx: mpsc::WeakUnboundedSender<OutFrame>,
    pub(super) mode: Mode,
}

impl WeakOutTx {
    pub(super) fn upgrade(&self) -> Option<OutTx> {
        Some(OutTx {
            tx: self.tx.upgrade()?,
            mode: self.mode.clone(),
        })
    }
}

impl OutTx {
    pub(super) fn new(tx: mpsc::UnboundedSender<OutFrame>, mode: Mode) -> Self {
        Self { tx, mode }
    }

    pub(super) fn send(&self, frame: OutFrame) -> Result<(), mpsc::error::SendError<OutFrame>> {
        self.tx.send(redact_frame(&self.mode, frame))
    }

    pub(super) fn downgrade(&self) -> WeakOutTx {
        WeakOutTx {
            tx: self.tx.downgrade(),
            mode: self.mode.clone(),
        }
    }
}

/// A frame queued for the single stdout writer task (see its own comment, above). Every RPC
/// `response`/`ack`/progress frame below is low-frequency and built as a `Value` — for those, `Value`
/// costs nothing extra: [`OutFanout::broadcast`] serializes it exactly once, and those bytes are what
/// is charged against every budget, recorded and sent to every connection. [`event_frame`] is the one high-frequency producer
/// (once per streamed model delta, far more often than any RPC frame) and carries its own pre-built
/// `String` instead, serialized straight from the `AgentEvent` in one pass — see its doc comment for
/// why going through `Value` first would cost a second full serialize on the hottest path this process
/// has.
/// `Raw` carries its already-serialized line as [`bytes::Bytes`], not `String`, so the two clones the
/// per-event fan-out path makes — the in-flight-turn recording ([`OutFanout::broadcast`]) and one per
/// extra attached sink — are refcount bumps rather than a full heap copy of the line. `event_frame`
/// still builds a `String` and hands its buffer over with a zero-copy `Bytes::from`.
#[derive(Clone)]
pub(crate) enum OutFrame {
    Value(Value),
    Raw(Bytes),
}

impl From<Value> for OutFrame {
    fn from(v: Value) -> Self {
        OutFrame::Value(v)
    }
}

impl OutFrame {
    /// Whether this frame answers one client's command (a `response` or `ack`) rather than reporting
    /// the session's turn. A reply belongs to the connection that asked: it is never replayed to one
    /// that attaches later (see [`OutFanout::turn`]).
    pub(super) fn is_reply(&self) -> bool {
        matches!(
            self,
            OutFrame::Value(v)
                if matches!(v.get("type").and_then(Value::as_str), Some("response" | "ack"))
        )
    }
}

/// The set of connections currently attached to a session, each a sink ([`OutSink`]) the session's
/// single writer task **broadcasts** every [`OutFrame`] to — so a phone and a TUI (or any N of the user's own
/// devices) on one session all render the same live stream, in the same order. Empty ⇒ detached, frames
/// dropped by design; the session keeps running, and the next connection to attach is caught up from
/// the committed history the fanout carries ([`OutFanout::add_with_catchup`] — see [`crate::serve_ws`]).
/// The supervisor [`add`](OutFanout::add)s a sink on attach and [`remove`](OutFanout::remove)s it on
/// disconnect. An `std::sync::Mutex` (not tokio's) is deliberate: it is held across non-`await` `send`s
/// into channels, and — once per attach, never on the event path — across serializing the catch-up
/// payload, which is precisely what makes "seed the history, then register the sink" atomic against a
/// concurrent `broadcast`.
/// One attached connection's output channel. The in-process **stdio** transport is a single trusted
/// local sink for the whole process life and stays **unbounded**: dropping it on a transient stdout
/// stall would sever the only client. A network transport (WebSocket/UDS/HTTP POST) registers a
/// [`ConnTx`] — buffered, but with a byte cap on its backlog, past which the connection is closed (see
/// there).
pub(crate) enum OutSink {
    Bounded(ConnTx),
    Unbounded(mpsc::UnboundedSender<OutFrame>),
}

impl OutSink {
    /// Non-blocking send of a live frame, already serialized. `Err` means the sink is dead: its receiver is gone, or — for
    /// a [`ConnTx`] — its backlog passed the cap and the connection is being closed.
    /// [`OutFanout::broadcast`] prunes it either way.
    pub(super) fn send_line(&self, line: Bytes) -> Result<(), ()> {
        match self {
            OutSink::Bounded(tx) => tx.send_line(line),
            OutSink::Unbounded(tx) => tx.send(OutFrame::Raw(line)).map_err(|_| ()),
        }
    }

    /// Queue a frame seeded on attach (the catch-up, the turn replay, kept views, the ending frame),
    /// outside the backlog cap: see [`ConnTx::seed`].
    pub(super) fn seed(&self, frame: OutFrame) {
        match self {
            OutSink::Bounded(tx) => tx.seed(frame),
            OutSink::Unbounded(tx) => {
                let _ = tx.send(frame);
            }
        }
    }
}

/// How many bytes of **live** output one network connection may have queued and not yet written to
/// its socket before it is dropped (see [`ConnTx`]).
///
/// Sized from what a healthy client actually falls behind by. The daemon's runtime is single-threaded
/// by default, so the session's writer moves everything one model read delivered — easily thousands of
/// deltas — into the channel before the connection's send task runs at all; a 3000-delta burst is
/// 252 KB queued at once (measured), against a client reading as fast as it can. 8 MiB is ~33× that,
/// and it still bounds what a wedged connection can pin.
pub(crate) const CONN_BACKLOG_MAX_BYTES: usize = 8 * 1024 * 1024;

/// A network connection's output channel: the session's broadcast on one end, the connection's send
/// task (which writes the socket) on the other.
///
/// **Buffered by bytes, then dropped — and a dropped connection is closed.**
///
/// - *Why buffer, and why count bytes.* The bound exists so that a client whose socket has stalled (a
///   locked phone, a dead tunnel) cannot make a still-running session buffer without limit. That is a
///   memory bound, so it is counted in bytes. It used to be a 1024-frame channel, which measured the
///   wrong thing: a single model read can carry more frames than that, and on the single-threaded
///   runtime they all land before the socket writer gets a turn, so a client that never stopped
///   reading was dropped mid-burst (`tests/serve_slow_client.rs`).
/// - *Why not backpressure.* Nothing upstream can wait. The agent's event sink is synchronous, the
///   session's one writer task feeds every attached connection (awaiting a slow one would stall the
///   others), and approvals, elicitations and app frames are broadcast straight from synchronous code
///   under the fanout's lock. A frame that does not fit cannot be delayed, only lost — and a lost frame
///   is a silent hole in the client's stream, so the connection goes instead.
/// - *Why the drop closes the socket.* Refusing a frame cancels [`dropped`](Self::dropped), which the
///   connection's send and read loops both watch: the socket closes (`1013 Try Again Later`) and the
///   client reconnects, to be caught up from history and the turn replay like any re-attach. Merely
///   unregistering the sink used to leave the socket open — still pinging, still running the client's
///   commands, never sending it another frame — so the client never knew to reconnect, and its
///   session counted as attached and was never reaped.
///
/// Frames seeded on attach are queued outside the cap ([`seed`](Self::seed)): they are bounded
/// already (by [`TURN_REPLAY_MAX_BYTES`] and the view budget), and refusing them would drop a client
/// for reconnecting.
#[derive(Clone)]
pub(crate) struct ConnTx {
    pub(super) tx: mpsc::UnboundedSender<(OutFrame, usize)>,
    pub(super) backlog: Arc<std::sync::atomic::AtomicUsize>,
    pub(super) dropped: CancellationToken,
    /// Whether this connection is also sent the `committed` marker (see [`OutFanout::broadcast_commit`]).
    /// Only the SSE stream asks: a WebSocket client's frames are exactly what they were.
    pub(super) marks: bool,
}

/// The receiving end of a [`ConnTx`], owned by the connection's send task.
pub(crate) struct ConnRx {
    pub(super) rx: mpsc::UnboundedReceiver<(OutFrame, usize)>,
    pub(super) backlog: Arc<std::sync::atomic::AtomicUsize>,
}

/// A network connection's output channel: see [`ConnTx`].
pub(crate) fn conn_channel() -> (ConnTx, ConnRx) {
    let (tx, rx) = mpsc::unbounded_channel();
    let backlog = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    (
        ConnTx {
            tx,
            backlog: backlog.clone(),
            dropped: CancellationToken::new(),
            marks: false,
        },
        ConnRx { rx, backlog },
    )
}

/// [`conn_channel`] for a connection that is also told, with a `committed` frame, whenever the
/// session's history commits: the event stream's resume ids.
pub(crate) fn conn_channel_marked() -> (ConnTx, ConnRx) {
    let (mut tx, rx) = conn_channel();
    tx.marks = true;
    (tx, rx)
}

impl ConnTx {
    /// Queue a live frame, charged against [`CONN_BACKLOG_MAX_BYTES`]. A frame that does not fit is
    /// refused and the connection dropped ([`Self::dropped`] fires); so is every frame after it.
    ///
    /// Charged its real size: a `Value` is serialized here, once, and those bytes are what is queued
    /// — a `get_messages` reply carries the whole transcript, and charging it a nominal size let a
    /// client pin many times the cap.
    pub(crate) fn try_send(&self, frame: OutFrame) -> Result<(), ()> {
        if self.dropped.is_cancelled() {
            return Err(());
        }
        let Some(line) = frame_to_line(frame) else {
            return Ok(());
        };
        self.send_line(line)
    }

    /// [`Self::try_send`] for a frame already serialized.
    pub(super) fn send_line(&self, line: Bytes) -> Result<(), ()> {
        if self.dropped.is_cancelled() {
            return Err(());
        }
        let len = line.len();
        let queued = self.backlog.fetch_add(len, Ordering::Relaxed);
        if queued + len > CONN_BACKLOG_MAX_BYTES
            || self.tx.send((OutFrame::Raw(line), len)).is_err()
        {
            self.dropped.cancel();
            return Err(());
        }
        Ok(())
    }

    /// Queue a frame outside the cap — for what [`OutFanout`] seeds a connection with on attach.
    pub(super) fn seed(&self, frame: OutFrame) {
        let _ = self.tx.send((frame, 0));
    }

    /// Fires once this connection has been dropped for falling too far behind: its socket must close.
    pub(crate) fn dropped(&self) -> CancellationToken {
        self.dropped.clone()
    }
}

impl ConnRx {
    /// The next queued frame, or `None` once every [`ConnTx`] is gone.
    pub(crate) async fn recv(&mut self) -> Option<OutFrame> {
        let (frame, charged) = self.rx.recv().await?;
        self.backlog.fetch_sub(charged, Ordering::Relaxed);
        Some(frame)
    }
}

#[derive(Default)]
pub(crate) struct OutFanout {
    pub(super) next_id: u64,
    pub(super) sinks: Vec<(u64, OutSink)>,
    /// The session's committed history, as of its last commit — what a connection attaching *now* has
    /// missed. Kept here, under the very same lock as `sinks`, on purpose: a connection attaching must
    /// be seeded with the history and registered as a sink **atomically**. Were these two behind
    /// separate locks, a message could commit in between — broadcast to the old sink set (which this
    /// connection is not yet in) *after* its history snapshot was taken — and would then appear in
    /// neither the catch-up nor the live stream. It would be dropped from that client's view entirely.
    /// See [`add_with_catchup`](Self::add_with_catchup).
    pub(super) history: Arc<Vec<agent_core::Message>>,
    /// Tree ids parallel to `history` (see `Persistence::active_ids`), so a catch-up tags messages
    /// exactly the way `get_messages` does. Empty when persistence tracks none (in-memory mode), which
    /// [`messages_payload`] already treats as "leave them untagged".
    pub(super) history_ids: Arc<Vec<String>>,
    /// Every frame this session has broadcast since the current run began — replayed to a connection
    /// that attaches mid-turn, right after its catch-up.
    ///
    /// Committed history alone is not enough to reconstruct what a client sees. The turn in flight has
    /// not committed yet, so it is nowhere in `history`, and the frames it has already emitted went out
    /// before this connection existed. Without this, a client re-attaching mid-turn would pick the turn
    /// up from the middle — a `tool_end` for a `tool_start` it never received, an assistant message
    /// missing its opening — which is the same "the transcript is wrong when I reconnect" symptom as
    /// delivering the backlog late.
    ///
    /// `history` deliberately does **not** advance across a run (only [`crate::serve::serve_session`]'s
    /// idle loop publishes it, and a run never returns there), which is what makes `history + turn`
    /// exactly reconstruct a never-dropped client's view with no overlap: the base is the transcript as
    /// of the run's start, and these are every frame since. Were `history` to advance at each mid-run
    /// checkpoint, a message would appear *both* in the catch-up and again in the replayed frames, and
    /// the client would render it twice.
    ///
    /// Replies are not recorded ([`OutFrame::is_reply`]): a `response` answers the client that sent
    /// the command, and replaying another client's answer — a `get_messages` reply is a whole
    /// transcript — to a connection that attaches later is wrong for it and grew this buffer by a
    /// transcript per call. Each frame is kept as the exact bytes sent, so its size is exact.
    pub(super) turn: Vec<Bytes>,
    /// Byte budget already spent on `turn`, against [`TURN_REPLAY_MAX_BYTES`].
    pub(super) turn_bytes: usize,
    /// `true` only between [`begin_turn`](Self::begin_turn) and [`end_turn`](Self::end_turn) — a run is
    /// in flight, so `broadcast` records what it sends. Idle, nothing is recorded and `turn` is empty.
    pub(super) recording: bool,
    /// The in-flight turn outgrew [`TURN_REPLAY_MAX_BYTES`] and its recording was dropped to bound
    /// memory. Surfaced on the catch-up as `turn_truncated: true` so a client that attaches knows its
    /// view of the current turn starts mid-way and must be reconciled with `get_messages` once the run
    /// ends — rather than silently rendering a turn with a hole in it.
    pub(super) turn_truncated: bool,
    /// The frame that ended the session, once one has ([`Self::end`]): why it is gone. Given to every
    /// sink that registers afterwards, because a connection can attach — its upgrade answered while
    /// the session was still starting — just after the session failed and broadcast it, and must
    /// still learn why it is being closed rather than see a bare close.
    pub(super) ended: Option<OutFrame>,
    /// The session's open MCP App views — each view's `mcp_app_open` and, once it lands, its
    /// `mcp_app_result` — replayed (marked `replay: true`) to every connection that attaches, so a
    /// client that reconnects can reopen a view and get its result. Kept apart from `turn` because a
    /// view outlives the turn that opened it: the turn recording is dropped when the run ends, and
    /// committed history carries the tool's text, not its view. Bounded by
    /// [`APP_VIEWS_MAX`]/[`APP_VIEWS_MAX_BYTES`], oldest view dropped first.
    pub(super) app_views: std::collections::VecDeque<AppViewFrames>,
    pub(super) app_views_bytes: usize,
    /// Bumped on every change to `app_views`, so the idle loop persists the store only when it moved.
    pub(super) app_views_version: u64,
}

/// One MCP App view's frames, for replay on attach (see [`OutFanout::app_views`]).
pub(super) struct AppViewFrames {
    pub(super) app_id: String,
    /// The server whose tool opened the view — what a restored view's bridge requests bind to.
    pub(super) server: String,
    /// The frames as sent — final JSON text, so their size is exact and a
    /// replay is a copy, not a re-serialization.
    pub(super) open: Bytes,
    pub(super) result: Option<Bytes>,
    pub(super) bytes: usize,
}

pub(super) fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|w| w == needle)
}

/// A kept app frame, marked `replay: true` for the connection it is replayed to.
pub(super) fn app_replay_frame(raw: &Bytes) -> OutFrame {
    let mut line = Vec::with_capacity(raw.len() + 16);
    line.extend_from_slice(b"{\"replay\":true,");
    line.extend_from_slice(raw.get(1..).unwrap_or_default());
    OutFrame::Raw(Bytes::from(line))
}

/// How many MCP App views a session keeps for replay. A view is one tool call's UI; a client
/// reconnecting needs the recent ones, not the whole session's history of them.
pub(super) const APP_VIEWS_MAX: usize = 16;

/// The byte budget for those views (a view's HTML is commonly a few hundred KiB, single-file bundled).
pub(super) const APP_VIEWS_MAX_BYTES: usize = 8 * 1024 * 1024;

// One view never exceeds the budget on its own (the agent refuses larger views outright).
pub(super) const _: () = assert!(crate::tools::mcp_apps::MAX_VIEW_BYTES < APP_VIEWS_MAX_BYTES);

/// How much of an in-flight turn's output a session will hold for replay to a connection that attaches
/// mid-turn (see [`OutFanout::turn`]). A cap is required, not optional: a single tool-heavy turn can emit
/// megabytes (a `read` of a large file, a chatty `bash`), this buffer is **per session**, and the daemon
/// multiplexes many — an unbounded recorder would turn one pathological turn into a per-session leak that
/// lives until the run ends. 4 MiB comfortably covers ordinary turns; past it the recording is dropped
/// and the catch-up says so.
pub(super) const TURN_REPLAY_MAX_BYTES: usize = 4 * 1024 * 1024;

impl OutFanout {
    /// Register a connection's sink; returns an id to [`remove`](Self::remove) it by on disconnect.
    pub(crate) fn add(&mut self, tx: OutSink) -> u64 {
        if let Some(ended) = &self.ended {
            tx.seed(ended.clone());
        }
        let id = self.next_id;
        self.next_id += 1;
        self.sinks.push((id, tx));
        id
    }

    /// The committed history and its parallel tree ids, for a caller that needs to answer from a
    /// snapshot rather than from `session` (a mid-run `get_messages` — the run holds `&mut session`).
    /// Both are `Arc`s, so this is two refcount bumps and the lock is released immediately.
    pub(crate) fn history_snapshot(&self) -> (Arc<Vec<agent_core::Message>>, Arc<Vec<String>>) {
        (self.history.clone(), self.history_ids.clone())
    }

    /// Publish the session's committed history, for the next connection to be caught up with. Called by
    /// the session whenever messages commit: at each mid-run durable checkpoint, and whenever an idle
    /// command changes them.
    ///
    /// The transcript itself costs nothing to publish — `Session::messages` is already an `Arc`, so it's
    /// a refcount bump. The ids are owned here (`Persistence::active_ids` only lends a slice), which is
    /// the one real copy; callers gate on the transcript's `Arc` pointer actually having changed, so an
    /// idle session never pays it.
    pub(crate) fn set_history(&mut self, messages: Arc<Vec<agent_core::Message>>, ids: &[String]) {
        self.history = messages;
        self.history_ids = Arc::new(ids.to_vec());
    }

    /// Register a connection's sink, **seeding it with the committed history first** so the new
    /// connection receives its backlog ahead of any live frame.
    ///
    /// This ordering is the whole point, and it is why the catch-up cannot be left to the client to ask
    /// for after connecting. Frames carry no sequence number, so delivery order is the only order a
    /// client has. A client that re-attaches to a session with a run in flight would otherwise start
    /// receiving that run's live output immediately — while its catch-up request raced along behind —
    /// and would render the tail of a turn whose earlier messages it had never seen.
    ///
    /// Seeding *before* [`add`](Self::add), under the lock [`broadcast`](Self::broadcast) also takes,
    /// makes the ordering structural rather than a race this happens to usually win: any concurrent
    /// broadcast either already ran (so its message is in `history`, or predates this connection) or
    /// blocks until the sink is registered — landing strictly after the catch-up already queued ahead
    /// of it.
    ///
    /// Nothing is sent when there is nothing to catch up on — a fresh session with no history and no run
    /// in flight has no backlog, so its first client gets no unsolicited frame.
    ///
    /// The lock is held across a `serde_json` serialization of the history here — the one place it is
    /// held across more than a channel send (see the type's own note). That is deliberate: it is what
    /// buys the atomicity above, it is bounded by the transcript length, and it happens once per
    /// *attach* (a client reconnecting), never on the hot event path.
    pub(crate) fn add_with_catchup(&mut self, tx: OutSink) -> u64 {
        let data = if !self.history.is_empty() || self.recording {
            messages_payload(&self.history, &self.history_ids, None).ok()
        } else {
            None
        };
        self.add_seeded(tx, data)
    }

    /// [`Self::add_with_catchup`] for a client that already holds the history up to the message `since`
    /// (the event stream's `Last-Event-ID`): the catch-up carries only what was committed after it.
    /// A `since` this session does not have is an `Err`, and nothing is registered: a stale id must
    /// not read as a working resume that silently re-sends everything (or nothing).
    pub(crate) fn add_with_catchup_since(
        &mut self,
        tx: OutSink,
        since: Option<&str>,
    ) -> Result<u64, String> {
        let data = if !self.history.is_empty() || self.recording || since.is_some() {
            Some(messages_payload(&self.history, &self.history_ids, since)?)
        } else {
            None
        };
        Ok(self.add_seeded(tx, data))
    }

    fn add_seeded(&mut self, tx: OutSink, data: Option<Value>) -> u64 {
        if let Some(data) = data {
            tx.seed(OutFrame::Value(json!({
                "type": "catchup",
                "data": data,
                // The turn in flight, if any, follows as replayed frames — unless it outgrew the
                // recorder, in which case say so rather than hand the client a turn with a hole in
                // it (see `turn_truncated`).
                "turn_in_flight": self.recording,
                "turn_truncated": self.turn_truncated,
            })));
        }
        // The session's open MCP App views, each in its place: a view opened by a tool call in the
        // turn being replayed goes right after that call's `tool_start`, so a renderer never sees a
        // view before the call it belongs to; a view from an earlier turn (whose `tool_start` is in
        // the catch-up's history) goes first. Everything seeded here is queued outside the
        // connection's backlog cap (`ConnTx::seed`), so none of it can be refused.
        let anchors: Vec<Option<usize>> = self
            .app_views
            .iter()
            .map(|view| {
                let needle = format!(
                    "\"kind\":\"tool_start\",\"id\":{}",
                    serde_json::to_string(&view.app_id).unwrap_or_default()
                );
                self.turn
                    .iter()
                    .position(|line| contains_bytes(line, needle.as_bytes()))
            })
            .collect();
        let send_view = |view: &AppViewFrames| {
            for frame in std::iter::once(&view.open).chain(view.result.as_ref()) {
                tx.seed(app_replay_frame(frame));
            }
        };
        for (view, anchor) in self.app_views.iter().zip(&anchors) {
            if anchor.is_none() {
                send_view(view);
            }
        }
        // Replay the in-flight turn *after* the catch-up and *before* the sink goes live: the base
        // transcript, then every frame since the run began, then the live stream — byte-identical to
        // what a client that never dropped has seen. All three inside this one critical section, so a
        // concurrent `broadcast` can neither interleave with the replay nor slip in between it and the
        // registration below.
        for (index, line) in self.turn.iter().enumerate() {
            tx.seed(OutFrame::Raw(line.clone()));
            for (view, anchor) in self.app_views.iter().zip(&anchors) {
                if *anchor == Some(index) {
                    send_view(view);
                }
            }
        }
        self.add(tx)
    }

    /// Broadcast an MCP App frame (`mcp_app_open` / `mcp_app_result`) and keep it for replay on
    /// attach (see [`Self::app_views`]). Not recorded into `turn`: the view store replays it, and
    /// replaying it from both would open the view twice.
    pub(crate) fn broadcast_app(&mut self, frame: Value) {
        let app_id = frame
            .get("app_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let kind = frame
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let Ok(line) = serde_json::to_vec(&frame) else {
            return;
        };
        let raw = OutFrame::Raw(Bytes::from(line));
        let OutFrame::Raw(bytes_line) = &raw else {
            return;
        };
        let bytes = bytes_line.len();
        match kind.as_str() {
            "mcp_app_open" => {
                self.app_views.retain(|v| v.app_id != app_id);
                if bytes <= APP_VIEWS_MAX_BYTES {
                    self.app_views.push_back(AppViewFrames {
                        app_id,
                        server: frame
                            .get("server")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        open: bytes_line.clone(),
                        result: None,
                        bytes,
                    });
                }
            }
            "mcp_app_result" => {
                if let Some(view) = self.app_views.iter_mut().find(|v| v.app_id == app_id) {
                    view.bytes += bytes;
                    view.result = Some(bytes_line.clone());
                }
            }
            // The view is gone for good: nothing to reopen on the next attach.
            "mcp_app_teardown" => self.app_views.retain(|v| v.app_id != app_id),
            _ => {}
        }
        self.trim_app_views();
        let recording = std::mem::replace(&mut self.recording, false);
        self.broadcast(raw);
        self.recording = recording;
    }

    /// Forget every kept MCP App view — the session they belonged to is no longer the one here.
    pub(crate) fn clear_app_views(&mut self) {
        self.app_views.clear();
        self.app_views_bytes = 0;
        self.app_views_version += 1;
    }

    /// Re-impose the store's bounds (newest views kept) and note that it changed.
    pub(super) fn trim_app_views(&mut self) {
        self.app_views_bytes = self.app_views.iter().map(|v| v.bytes).sum();
        while self.app_views.len() > APP_VIEWS_MAX
            || (self.app_views_bytes > APP_VIEWS_MAX_BYTES && self.app_views.len() > 1)
        {
            if let Some(dropped) = self.app_views.pop_front() {
                self.app_views_bytes -= dropped.bytes;
            }
        }
        self.app_views_version += 1;
    }

    /// The replay store's version — see `app_views_version`.
    pub(crate) fn app_views_version(&self) -> u64 {
        self.app_views_version
    }

    /// The kept views' `(app_id, server)` — what a fresh `set_mcp_apps` seeds its open-view map with,
    /// so a view restored after a restart can still call back.
    pub(crate) fn app_view_servers(&self) -> Vec<(String, String)> {
        self.app_views
            .iter()
            .map(|v| (v.app_id.clone(), v.server.clone()))
            .collect()
    }

    /// The replay store as the views sidecar records it (`None` when empty): one object per view,
    /// oldest first, carrying the frames exactly as sent.
    pub(crate) fn app_views_snapshot(&self) -> Option<Vec<u8>> {
        if self.app_views.is_empty() {
            return None;
        }
        let parse = |b: &Bytes| serde_json::from_slice::<Value>(b).unwrap_or(Value::Null);
        let views: Vec<Value> = self
            .app_views
            .iter()
            .map(|v| {
                json!({
                    "app_id": v.app_id,
                    "server": v.server,
                    "open": parse(&v.open),
                    "result": v.result.as_ref().map(parse),
                })
            })
            .collect();
        serde_json::to_vec(&views).ok()
    }

    /// Replace the replay store with a views sidecar's contents (see [`Self::app_views_snapshot`]),
    /// re-bounded by the same budget. Malformed entries are skipped.
    pub(crate) fn restore_app_views(&mut self, bytes: &[u8]) {
        let Ok(Value::Array(views)) = serde_json::from_slice::<Value>(bytes) else {
            return;
        };
        self.app_views.clear();
        for v in views {
            let (Some(app_id), Some(open)) = (v["app_id"].as_str(), v.get("open")) else {
                continue;
            };
            if open["type"] != "mcp_app_open" {
                continue;
            }
            let Ok(open) = serde_json::to_vec(open) else {
                continue;
            };
            let result = v
                .get("result")
                .filter(|r| !r.is_null())
                .and_then(|r| serde_json::to_vec(r).ok())
                .map(Bytes::from);
            let bytes = open.len() + result.as_ref().map_or(0, Bytes::len);
            self.app_views.push_back(AppViewFrames {
                app_id: app_id.to_owned(),
                server: v["server"].as_str().unwrap_or_default().to_owned(),
                open: Bytes::from(open),
                result,
                bytes,
            });
        }
        self.trim_app_views();
        // A connection may already be attached (the daemon attaches before the session's first
        // idle pass, where this runs): replay the restored views to it now, as an attach would.
        for view in &self.app_views {
            for frame in std::iter::once(&view.open).chain(view.result.as_ref()) {
                for (_, tx) in &self.sinks {
                    tx.seed(app_replay_frame(frame));
                }
            }
        }
    }

    /// A run is starting: record what it broadcasts, for replay to anyone who attaches mid-turn.
    ///
    /// Called once per `prompt`, not per auto-retry attempt — a client that never dropped saw every
    /// attempt's frames, including the `auto_retry_start` notices between them, so a reconnecting one is
    /// owed the same.
    pub(crate) fn begin_turn(&mut self) {
        self.turn.clear();
        self.turn_bytes = 0;
        self.turn_truncated = false;
        self.recording = true;
    }

    /// The run is over and its messages have committed into `history` — drop the recording, which would
    /// now duplicate what the catch-up already carries.
    ///
    /// Idempotent, and called by the idle loop on every pass (including when no run ever started), which
    /// is also what releases the memory a tool-heavy turn's recording held: the buffer is freed the
    /// moment the session goes idle, not kept until the next run overwrites it.
    pub(crate) fn end_turn(&mut self) {
        self.turn = Vec::new();
        self.turn_bytes = 0;
        self.turn_truncated = false;
        self.recording = false;
    }

    /// Drop the sink registered under `id` (a disconnected connection). Idempotent.
    pub(crate) fn remove(&mut self, id: u64) {
        self.sinks.retain(|(i, _)| *i != id);
    }

    /// Whether no client is currently attached. The stdio transport registers exactly one sink for the
    /// life of the process, so this is only ever `false` there; a WebSocket/UDS session whose clients
    /// have all detached reports `true`. [`crate::approval`] checks this before posing a question that
    /// nobody would ever see.
    pub(crate) fn is_empty(&self) -> bool {
        self.sinks.is_empty()
    }

    /// Send `frame` to every attached sink, pruning any whose receiver has gone away. The single-sink
    /// case (one connection — the common case) moves the frame with **no clone**, preserving the hot
    /// event path's zero-copy behavior; only 2+ attached connections pay a per-sink clone.
    ///
    /// While a run is in flight the frame is also recorded for replay (see [`Self::turn`]) — that clone
    /// is the one allocation this path gains, and only for the duration of a turn. Note the recording
    /// happens even with **zero** sinks attached: a fully detached session still has to be able to catch
    /// the next client up on the turn it ran while nobody was watching, which is the whole point of a
    /// session outliving its connection.
    /// Broadcast the frame that ends the session (why it is gone), and keep it for any connection
    /// that registers afterwards (see [`Self::ended`]).
    pub(crate) fn end(&mut self, frame: OutFrame) {
        self.ended = Some(frame.clone());
        self.broadcast(frame);
    }

    /// The session's history just committed: tell the connections that asked (the event stream) the
    /// id of the newest committed message, which is what they resume from. Not recorded for replay
    /// (it is about the commit, and the replay ends at it), and not sent to a WebSocket client.
    pub(crate) fn broadcast_commit(&mut self) {
        let Some(leaf) = self.history_ids.last() else {
            return;
        };
        let Some(line) = frame_to_line(OutFrame::Value(
            json!({ "type": "committed", "leaf_id": leaf }),
        )) else {
            return;
        };
        self.sinks.retain(|(_, tx)| match tx {
            OutSink::Bounded(c) if c.marks => tx.send_line(line.clone()).is_ok(),
            _ => true,
        });
    }

    pub(crate) fn broadcast(&mut self, frame: OutFrame) {
        let reply = frame.is_reply();
        // Serialized once, here: these bytes are what every budget is charged, what the recording
        // keeps and what each connection is sent (a refcount bump per sink, not a copy).
        let Some(line) = frame_to_line(frame) else {
            return;
        };
        if self.recording && !self.turn_truncated && !reply {
            let len = line.len();
            if self.turn_bytes + len > TURN_REPLAY_MAX_BYTES {
                // Past the budget: drop the whole recording (freeing it now, when the memory pressure is
                // real) and mark the turn truncated. A client attaching from here on is told its view of
                // this turn is partial rather than being handed one with a silent hole in it.
                self.turn = Vec::new();
                self.turn_bytes = 0;
                self.turn_truncated = true;
            } else {
                self.turn_bytes += len;
                self.turn.push(line.clone());
            }
        }
        match self.sinks.as_slice() {
            [] => {}
            [(_, tx)] => {
                if tx.send_line(line).is_err() {
                    self.sinks.clear();
                }
            }
            _ => self
                .sinks
                .retain(|(_, tx)| tx.send_line(line.clone()).is_ok()),
        }
    }
}

/// Shared handle to a session's attached-connection set (see [`OutFanout`]).
pub(crate) type SharedOutConn = Arc<std::sync::Mutex<OutFanout>>;

#[cfg(test)]
mod tests {
    use super::*;

    /// A views sidecar of `n` views, oldest first, each opening `html_bytes` of HTML.
    fn views_sidecar(n: usize, html_bytes: usize) -> Vec<u8> {
        let html = "h".repeat(html_bytes);
        let views: Vec<Value> = (0..n)
            .map(|i| {
                json!({
                    "app_id": format!("app-{i:02}"),
                    "server": "s",
                    "open": { "type": "mcp_app_open", "app_id": format!("app-{i:02}"), "html": html },
                    "result": { "type": "mcp_app_result", "app_id": format!("app-{i:02}") },
                })
            })
            .collect();
        serde_json::to_vec(&views).unwrap()
    }

    fn kept_ids(fanout: &OutFanout) -> Vec<String> {
        fanout
            .app_view_servers()
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    #[test]
    fn a_restored_views_sidecar_is_re_bounded_to_the_view_count() {
        // A sidecar written by an older binary, or edited on disk, may hold more than the store
        // keeps: the restore re-imposes the bound, newest views kept.
        let mut fanout = OutFanout::default();
        fanout.restore_app_views(&views_sidecar(APP_VIEWS_MAX + 4, 16));
        let want: Vec<String> = (4..APP_VIEWS_MAX + 4)
            .map(|i| format!("app-{i:02}"))
            .collect();
        assert_eq!(kept_ids(&fanout), want);
        // And what it would write back is the bounded store, not the oversized input.
        let snapshot: Vec<Value> =
            serde_json::from_slice(&fanout.app_views_snapshot().unwrap()).unwrap();
        assert_eq!(snapshot.len(), APP_VIEWS_MAX);
    }

    #[test]
    fn a_restored_views_sidecar_is_re_bounded_to_the_byte_budget() {
        // Four 3 MiB views are 12 MiB: over the 8 MiB budget, so only the newest two survive.
        let mut fanout = OutFanout::default();
        fanout.restore_app_views(&views_sidecar(4, 3 * 1024 * 1024));
        assert_eq!(kept_ids(&fanout), ["app-02", "app-03"]);
        let snapshot = fanout.app_views_snapshot().unwrap();
        assert!(
            snapshot.len() <= APP_VIEWS_MAX_BYTES,
            "{} bytes kept",
            snapshot.len()
        );
    }

    /// A live frame of `len` bytes, numbered so a receiver can check order.
    fn sized_frame(i: usize, len: usize) -> OutFrame {
        let mut line = format!("{i:08}").into_bytes();
        line.resize(len, b'x');
        OutFrame::Raw(Bytes::from(line))
    }

    fn frame_number(frame: &OutFrame) -> usize {
        let OutFrame::Raw(line) = frame else {
            panic!("not a raw frame")
        };
        std::str::from_utf8(&line[..8]).unwrap().parse().unwrap()
    }

    #[test]
    fn a_connection_past_its_backlog_cap_is_dropped_and_told_to_close() {
        let (tx, mut rx) = conn_channel();
        let dropped = tx.dropped();
        let mut fanout = OutFanout::default();
        fanout.add(OutSink::Bounded(tx));
        let frame_len = 64 * 1024;
        let fits = CONN_BACKLOG_MAX_BYTES / frame_len;
        for i in 0..fits {
            fanout.broadcast(sized_frame(i, frame_len));
        }
        assert!(
            !dropped.is_cancelled() && !fanout.is_empty(),
            "dropped at the cap, not past it"
        );
        // One frame more than fits: the connection goes, and is told to close its socket — rather
        // than being silently unregistered while it stays open.
        fanout.broadcast(sized_frame(fits, frame_len));
        assert!(dropped.is_cancelled());
        assert!(fanout.is_empty());
        // What it was sent is whole and in order, with nothing after the refused frame.
        let mut got = Vec::new();
        while let Ok((frame, _)) = rx.rx.try_recv() {
            got.push(frame_number(&frame));
        }
        assert_eq!(got, (0..fits).collect::<Vec<_>>());
    }

    #[test]
    fn a_connection_that_keeps_draining_is_never_dropped() {
        // Many times the cap in total, but never more than a little of it queued at once.
        let (tx, mut rx) = conn_channel();
        let dropped = tx.dropped();
        let mut fanout = OutFanout::default();
        fanout.add(OutSink::Bounded(tx));
        let frame_len = 64 * 1024;
        let total = 4 * CONN_BACKLOG_MAX_BYTES / frame_len;
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        for i in 0..total {
            fanout.broadcast(sized_frame(i, frame_len));
            if i % 16 == 15 {
                for _ in 0..16 {
                    rt.block_on(rx.recv()).unwrap();
                }
            }
        }
        assert!(!dropped.is_cancelled() && !fanout.is_empty());
    }

    /// A client calling `get_messages` on a large transcript during a run: every reply is charged its
    /// real size against each connection's backlog, and none is recorded for replay — the turn
    /// recording holds the turn's own frames only. (Replies used to be charged a flat 256 bytes and
    /// recorded: fifty ~1 MiB replies cost 12,800 bytes of budget, the replay buffer grew by a
    /// transcript per call, and a stalled connection pinned all of it.)
    #[test]
    fn a_large_reply_is_charged_its_size_and_is_not_replayed() {
        let transcript = "t".repeat(1024 * 1024);
        let reply = || {
            response(
                Some("g".into()),
                "get_messages",
                true,
                Some(json!({ "messages": [transcript] })),
                None,
            )
        };
        let mut fanout = OutFanout::default();
        fanout.begin_turn();
        let (healthy, mut healthy_rx) = conn_channel();
        let (stalled, _stalled_rx) = conn_channel();
        let healthy_dropped = healthy.dropped();
        let stalled_dropped = stalled.dropped();
        fanout.add(OutSink::Bounded(healthy));
        fanout.add(OutSink::Bounded(stalled));
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        for i in 0..50 {
            fanout.broadcast(reply());
            // The healthy client reads each reply as it comes, and is charged all of it.
            let OutFrame::Raw(line) = rt.block_on(healthy_rx.recv()).unwrap() else {
                panic!("a live frame is sent as its serialized bytes")
            };
            assert!(line.len() > transcript.len());
            assert_eq!(healthy_rx.backlog.load(Ordering::Relaxed), 0);
            if i == CONN_BACKLOG_MAX_BYTES / transcript.len() {
                assert!(
                    stalled_dropped.is_cancelled(),
                    "a connection that reads nothing is dropped once the replies pass its cap"
                );
            }
        }
        assert!(!healthy_dropped.is_cancelled());
        assert!(stalled_dropped.is_cancelled());
        assert!(fanout.turn.is_empty(), "no reply is recorded for replay");
        assert_eq!(fanout.turn_bytes, 0);
        assert!(!fanout.turn_truncated);

        // The turn's own frames still are, charged exactly.
        fanout.broadcast(sized_frame(0, 1000));
        assert_eq!(fanout.turn.len(), 1);
        assert_eq!(fanout.turn_bytes, 1000);
    }

    #[test]
    fn frames_seeded_on_attach_do_not_count_against_the_backlog_cap() {
        // A reconnect is seeded with the turn replay and kept views — together more than the cap can
        // be — and must not be dropped for it.
        let mut fanout = OutFanout::default();
        fanout.begin_turn();
        let frame_len = 64 * 1024;
        for i in 0..TURN_REPLAY_MAX_BYTES / frame_len {
            fanout.broadcast(sized_frame(i, frame_len));
        }
        let (tx, _rx) = conn_channel();
        let dropped = tx.dropped();
        for i in 0..2 * CONN_BACKLOG_MAX_BYTES / frame_len {
            OutSink::Bounded(tx.clone()).seed(sized_frame(i, frame_len));
        }
        fanout.add_with_catchup(OutSink::Bounded(tx));
        fanout.broadcast(sized_frame(0, 256));
        assert!(!dropped.is_cancelled() && !fanout.is_empty());
    }
}
