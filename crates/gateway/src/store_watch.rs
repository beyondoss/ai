//! slipstream control-plane watchers — the gateway's **only** use of NATS.
//!
//! Seeds a sparse per-tenant set at boot, then streams deltas. **Fail-open**: a NATS blip keeps the
//! last-known set (we never clear), so an outage degrades to a stale set, not "reject everything".
//! Auth and pool/signing keys come from config, so they're unaffected by NATS being down — only
//! spend/fraud enforcement goes stale.
//!
//! Two sets are watched, both per-tenant and both sparse, differing only in prefix and payload: the
//! **deny-set** (`blackhole.`, enforcement) and the **capture-set** (`aicapture.`, payload logging).
//! The seed → watch → batch-apply → reconnect loop below is written **once**, over the [`WatchedSet`]
//! trait, and instantiated per set. That loop carries several non-obvious correctness properties —
//! `is_resumable`'s revision-0 trap, the scan→subscribe race, batched `rcu`, backoff crediting — and
//! a copied second version would be a standing invitation for the two to drift apart. Each set gets
//! its own [`WatcherService`] (hence its own NATS connection, cursor, and reconnect loop), so a
//! capture-set problem cannot disturb deny enforcement.
//!
//! Seeding has two modes, chosen by `config.snapshot_path`:
//!
//! - **Unset (ephemeral, e.g. Fargate):** scan `blackhole.*` from NATS on first connect. The resume
//!   revision is kept *in memory* across reconnects, so a NATS blip resumes the watch from where it
//!   left off (gap-free) rather than re-scanning.
//! - **Set (edge/tunnel, durable disk):** load slipstream's on-disk snapshot (entries + a saved
//!   watch cursor), seed from it, and resume the watch from that cursor — a restart skips the scan
//!   and enforces immediately, even before NATS reconnects. Every applied delta is appended back to
//!   the snapshot so the file tracks the live set.
//!
//! Either way the watch *tries* to resume from a **revision** (`watch_prefix_from`) rather than a
//! bare `watch_prefix`: the latter uses NATS `DeliverPolicy::New` (no replay), so a deny entry
//! written in the window between seeding and the subscription attaching would be silently lost.
//! Resuming from the seeded revision closes that window with no gap and no double-apply (it starts
//! strictly after the seeded revision).
//!
//! The one seed that yields *no* resume point is an empty scan: its baseline is revision 0, and
//! slipstream honours a cursor only when `rev > 0` — at 0 `watch_prefix_from` quietly degrades to
//! `watch_prefix`, i.e. `DeliverPolicy::New`. We can't stop that degradation, so we refuse to treat
//! it as seeded (see `is_resumable`): the watch runs replay-less for this connection, and the *next*
//! connect rescans instead of latching a replay-less watch for the life of the process. Same for
//! `CursorExpired` (the backend compacted past the cursor) — we drop back to a fresh scan, which
//! re-establishes a valid baseline.
//!
//! Runs as a Pingora `BackgroundService` so the NATS client is created on the serving runtime
//! (async-nats ties its tasks to the runtime it's built on; connecting earlier would break it).

use crate::capture::{self, CaptureSet};
use crate::deny::{self, DenySet};
use crate::state::GatewayState;
use arc_swap::ArcSwap;
use async_trait::async_trait;
use pingora_core::server::ShutdownWatch;
use pingora_core::services::background::BackgroundService;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use store::snapshot::SnapshotWriter;
use store::{
    Connection, KvEntry, KvError, KvStore, KvUpdate, NatsConnection, NatsConnectionConfig,
    StoreConfig, WatchCursor,
};
use tokio::time::Instant;
use tracing::{error, info, warn};

/// Depth of the channel the watch task feeds, and therefore the most deltas one `recv_many` can
/// drain into a batch — the queue is the batch, so there's nothing to gain from a second bound.
const WATCH_CHANNEL_CAPACITY: usize = 256;

/// Compact the on-disk snapshot once it grows past this many bytes of appended deltas. The deny-set
/// is low-churn, so this is rarely hit; it just bounds the log if a tenant flaps.
const SNAPSHOT_COMPACT_THRESHOLD: u64 = 1024 * 1024;

/// Reconnect backoff bounds: start at 1s, double to a 30s ceiling. Generous enough to stop log spam
/// during a long NATS outage, tight enough that recovery is near-immediate once it returns. The cap
/// doubles as the "was that connection productive?" threshold — see `ReconnectBackoff`.
const RECONNECT_BACKOFF_BASE: Duration = Duration::from_secs(1);
const RECONNECT_BACKOFF_MAX: Duration = Duration::from_secs(30);

/// One watched control-plane set: which prefix it lives under, how to build it, and where to publish
/// it. Implemented once per set ([`Deny`], and the capture-set), then driven by [`watch_set`].
///
/// This exists so the seed → watch → batch-apply → reconnect loop is written **once**. Every method
/// here is a place the two sets genuinely differ; everything else about the loop — and in particular
/// every one of its subtle correctness properties — is shared by construction rather than by two
/// copies staying in sync.
///
/// Not object-safe (generic method, associated const), and deliberately so: each set is instantiated
/// statically at boot, so there's nothing to gain from dynamic dispatch on a path this cold.
pub trait WatchedSet: Send + Sync + 'static {
    /// The in-memory set this watcher publishes.
    type Set: Send + Sync + 'static;

    /// KV key prefix this set is scanned and watched under.
    const PREFIX: &'static str;

    /// Noun for this watcher's log lines ("deny-set", "capture-set") — so an oncall reading a
    /// `"…watch exited; reconnecting"` line can tell which of the two it came from.
    const NOUN: &'static str;

    /// Build the set from scanned or snapshot-loaded entries, dropping any malformed key/value.
    ///
    /// `state` is threaded through because a set's *values* can depend on boot config — the
    /// capture-set resolves its per-tenant defaults from `AiConfig` — even though its keys never do.
    /// The deny-set ignores it.
    fn from_entries<'a>(
        state: &GatewayState,
        entries: impl Iterator<Item = &'a KvEntry>,
    ) -> Self::Set;

    /// Fold a batch of watched deltas into a new set. See [`apply_batch`] for why this is per-batch
    /// rather than per-delta.
    fn apply_batch(state: &GatewayState, cur: &Self::Set, updates: &[KvUpdate]) -> Self::Set;

    /// Cardinality, for the size gauge and the seed log line.
    fn len(set: &Self::Set) -> usize;

    /// Where the live set is published for the request path to read.
    fn slot(state: &GatewayState) -> &ArcSwap<Self::Set>;

    /// Publish the current cardinality to this set's own gauge.
    fn record_size(state: &GatewayState, len: usize);

    /// Publish this watcher's NATS connectivity to its **own** gauge.
    ///
    /// Per-set rather than shared: two independent watchers writing one `nats_connected` gauge would
    /// race, and the reading would mean "whichever reconnected most recently", which is exactly the
    /// wrong answer during the partial outage you'd be looking at it for.
    fn record_connected(state: &GatewayState, connected: bool);

    /// Path to this set's on-disk snapshot, or `None` to run snapshot-less.
    ///
    /// The snapshot exists so *enforcement* survives a cold start before NATS reconnects. A set that
    /// isn't enforcement (the capture-set) fails open — capture simply stays off until the first
    /// scan lands — so it has nothing to gain from durable seeding and returns `None`, which makes
    /// every snapshot path below inert without needing to special-case it.
    fn snapshot_path(state: &GatewayState) -> Option<String>;
}

/// The deny-set: spend/fraud holds written by the control plane under `blackhole.{tenant}`.
pub struct Deny;

impl WatchedSet for Deny {
    type Set = DenySet;

    const PREFIX: &'static str = "blackhole.";
    const NOUN: &'static str = "deny-set";

    fn from_entries<'a>(
        _state: &GatewayState,
        entries: impl Iterator<Item = &'a KvEntry>,
    ) -> DenySet {
        denyset_from_entries(entries)
    }

    fn apply_batch(_state: &GatewayState, cur: &DenySet, updates: &[KvUpdate]) -> DenySet {
        apply_batch(cur, updates)
    }

    fn len(set: &DenySet) -> usize {
        set.len()
    }

    fn slot(state: &GatewayState) -> &ArcSwap<DenySet> {
        &state.deny
    }

    fn record_size(state: &GatewayState, len: usize) {
        state.metrics.deny_set_size.set(len as i64);
    }

    fn record_connected(state: &GatewayState, connected: bool) {
        state.metrics.nats_connected.set(i64::from(connected));
    }

    fn snapshot_path(state: &GatewayState) -> Option<String> {
        state.config.snapshot_path.clone()
    }
}

/// The capture-set: payload-logging opt-ins written by the control plane under `aicapture.{tenant}`.
pub struct Capture;

impl WatchedSet for Capture {
    type Set = CaptureSet;

    const PREFIX: &'static str = "aicapture.";
    const NOUN: &'static str = "capture-set";

    fn from_entries<'a>(
        state: &GatewayState,
        entries: impl Iterator<Item = &'a KvEntry>,
    ) -> CaptureSet {
        let defaults = state.capture_defaults;
        entries
            .filter_map(|e| {
                Some((
                    capture::parse_key(&e.key)?,
                    capture::parse_rule(&e.value, defaults),
                ))
            })
            .collect()
    }

    fn apply_batch(state: &GatewayState, cur: &CaptureSet, updates: &[KvUpdate]) -> CaptureSet {
        let defaults = state.capture_defaults;
        let mut set = cur.clone();
        for update in updates {
            match update {
                KvUpdate::Put(e) => {
                    if let Some(t) = capture::parse_key(&e.key) {
                        set.insert(t, capture::parse_rule(&e.value, defaults));
                    }
                }
                // Delete/Purge = capture off (explicit delete, or the TTL on a time-boxed
                // "capture tenant 42 for an hour" entry expiring).
                KvUpdate::Delete { key, .. } | KvUpdate::Purge { key, .. } => {
                    if let Some(t) = capture::parse_key(key) {
                        set.remove(t);
                    }
                }
            }
        }
        set
    }

    fn len(set: &CaptureSet) -> usize {
        set.len()
    }

    fn slot(state: &GatewayState) -> &ArcSwap<CaptureSet> {
        &state.capture
    }

    fn record_size(state: &GatewayState, len: usize) {
        state.metrics.capture_set_size.set(len as i64);
    }

    fn record_connected(state: &GatewayState, connected: bool) {
        state
            .metrics
            .capture_nats_connected
            .set(i64::from(connected));
    }

    /// No snapshot. The deny-set keeps one so *enforcement* survives a cold start before NATS
    /// reconnects; capture is not enforcement, and "captured nothing for the first few seconds after
    /// a restart" is a non-event. Returning `None` makes every snapshot path in this module inert
    /// for this set without a single `if` anywhere else.
    fn snapshot_path(_state: &GatewayState) -> Option<String> {
        None
    }
}

/// Reconnect backoff for the watch loop: 1s doubling to a 30s cap. A fixed 2s retry hammered the log
/// at a constant rate through a long outage (minutes to hours), burying other signals during the very
/// incident an oncall is reading these logs for. The gateway serves correctly on the stale set
/// throughout — this is purely about log volume and not pointlessly spinning on a down NATS.
///
/// The rule that matters is *when* the delay resets. It resets only after a connection that proved
/// **productive** — one whose watch survived at least [`RECONNECT_BACKOFF_MAX`]. Resetting on a
/// successful `connect()` instead (which is what this used to do) is a trap: NATS happily accepts
/// connections while the *watch* fails every single time — bucket missing, scan permission denied,
/// store hands back no watcher, consumer-create refused. Each iteration then reset the delay to the
/// base before `watch_deny` failed, undoing the doubling that follows, and the "backoff" pinned
/// itself at 1s: ~86k connects (TLS handshake + JetStream KV lookup + scan) and ~86k `warn!` lines a
/// day — exactly the log flood the backoff exists to prevent.
///
/// Kept as a tiny state machine so that rule is testable without a NATS server.
struct ReconnectBackoff(Duration);

impl ReconnectBackoff {
    fn new() -> Self {
        Self(RECONNECT_BACKOFF_BASE)
    }

    /// How long to wait before the next connect attempt.
    fn delay(&self) -> Duration {
        self.0
    }

    /// Double the delay, capped. Called after every attempt that didn't earn a reset.
    fn grow(&mut self) {
        self.0 = (self.0 * 2).min(RECONNECT_BACKOFF_MAX);
    }

    /// Credit a finished watch: a connection whose watch outlived the cap was doing real work
    /// (streaming deltas, or idling on a healthy subscription), so its successor starts fresh.
    /// Anything shorter is flapping, and flapping keeps backing off.
    fn credit(&mut self, watch_ran_for: Duration) {
        if watch_ran_for >= RECONNECT_BACKOFF_MAX {
            self.0 = RECONNECT_BACKOFF_BASE;
        }
    }
}

/// Drives one [`WatchedSet`] for the life of the process: seed, watch, reconnect.
///
/// One service per set, each with its own NATS connection and reconnect loop. The extra connection
/// is cheap next to the isolation it buys: a capture-set scan that keeps failing backs off on its
/// own schedule and cannot slow, stall, or reseed deny enforcement.
pub struct WatcherService<W: WatchedSet> {
    state: Arc<GatewayState>,
    _set: PhantomData<W>,
}

impl<W: WatchedSet> WatcherService<W> {
    pub fn new(state: Arc<GatewayState>) -> Self {
        Self {
            state,
            _set: PhantomData,
        }
    }
}

#[async_trait]
impl<W: WatchedSet> BackgroundService for WatcherService<W> {
    async fn start(&self, mut shutdown: ShutdownWatch) {
        // Resume position + on-disk snapshot writer persist across reconnects: a NATS blip resumes
        // the watch from `cursor` instead of re-scanning, and `seeded` stays true so we don't reseed.
        let mut cursor = WatchCursor::none();
        let mut writer: Option<SnapshotWriter> = None;
        let mut seeded = false;

        if let Some(path) = W::snapshot_path(&self.state) {
            let path = PathBuf::from(path);
            // Snapshot I/O is synchronous (whole-file read/rewrite) — offload it so we never stall
            // the serving runtime this BackgroundService shares with the proxy.
            let load_path = path.clone();
            match tokio::task::spawn_blocking(move || store::snapshot::load(&load_path)).await {
                Ok(Ok(Some(snap))) => {
                    let set = W::from_entries(&self.state, snap.entries.values());
                    let count = W::len(&set);
                    info!(set = W::NOUN, count, "seeded from on-disk snapshot");
                    W::record_size(&self.state, count);
                    W::slot(&self.state).store(Arc::new(set));
                    // A snapshot without a *resumable* cursor can't safely resume (a bare watch
                    // would race), so only treat it as seeded when it carries a real resume point;
                    // otherwise fall through to a NATS scan on connect. Note this must test
                    // `is_resumable`, not `!is_none()`: a snapshot checkpointed after an empty scan
                    // carries revision 0, which is a *present* cursor that slipstream still can't
                    // resume from.
                    if is_resumable(&snap.cursor) {
                        cursor = snap.cursor;
                        seeded = true;
                    }
                }
                Ok(Ok(None)) => info!("no on-disk snapshot yet; will seed from a NATS scan"),
                Ok(Err(e)) => warn!(error = %e, "snapshot load failed; will seed from a NATS scan"),
                Err(e) => warn!(error = %e, "snapshot load task panicked; seeding from NATS"),
            }
            let open_path = path.clone();
            match tokio::task::spawn_blocking(move || {
                SnapshotWriter::open(&open_path, SNAPSHOT_COMPACT_THRESHOLD)
            })
            .await
            {
                Ok(Ok(w)) => writer = Some(w),
                Ok(Err(e)) => warn!(error = %e, "snapshot writer open failed; running without it"),
                Err(e) => warn!(error = %e, "snapshot writer open task panicked"),
            }
        }

        let mut backoff = ReconnectBackoff::new();
        loop {
            // Connect, but bail immediately if Pingora signals shutdown mid-connect (e.g. NATS is
            // down and `connect` is retrying its own backoff) rather than blocking teardown.
            let store = tokio::select! {
                _ = shutdown.changed() => {
                    info!(
                        set = W::NOUN,
                        in_flight = self.state.metrics.requests_in_flight.get(),
                        "shutdown signaled; watcher exiting"
                    );
                    return;
                }
                outcome = connect(&self.state) => match outcome {
                    Ok(store) => store,
                    Err(e) => {
                        W::record_connected(&self.state, false);
                        error!(error = %e, backoff_secs = backoff.delay().as_secs(), "slipstream connect failed; retrying");
                        // Reconnect backoff, also interruptible by shutdown.
                        tokio::select! {
                            _ = shutdown.changed() => return,
                            _ = tokio::time::sleep(backoff.delay()) => {
                                backoff.grow();
                                continue;
                            }
                        }
                    }
                },
            };

            W::record_connected(&self.state, true);
            info!(set = W::NOUN, "slipstream connected; watching");
            // Time the watch, not the connect: only a watch that ran long enough to be doing real
            // work earns a backoff reset (see `ReconnectBackoff::credit`).
            let started = Instant::now();
            // `watch_set` returns `true` when it exited because shutdown was signaled — stop the
            // reconnect loop cleanly instead of trying to reconnect a shutting-down process.
            if watch_set::<W>(
                &self.state,
                store,
                &mut cursor,
                &mut writer,
                &mut seeded,
                &mut shutdown,
            )
            .await
            {
                info!(set = W::NOUN, "shutdown signaled; watcher exiting");
                return;
            }
            let watched_for = started.elapsed();
            backoff.credit(watched_for);
            W::record_connected(&self.state, false);
            warn!(
                set = W::NOUN,
                watched_secs = watched_for.as_secs(),
                backoff_secs = backoff.delay().as_secs(),
                "watch exited; reconnecting"
            );
            tokio::select! {
                _ = shutdown.changed() => return,
                _ = tokio::time::sleep(backoff.delay()) => backoff.grow(),
            }
        }
    }
}

/// Is this cursor a real resume point?
///
/// Only if it carries a revision **greater than zero**. slipstream's `watch_prefix_from` honours a
/// cursor exactly when `as_u64()` yields `Some(rev)` with `rev > 0`; anything else — an unknown
/// token, or the revision 0 an empty scan produces — silently falls back to `watch_prefix`, i.e.
/// NATS `DeliverPolicy::New`, a watch with no replay. That fallback is invisible from here (no
/// error, no log), so we have to make the same judgement ourselves: a cursor that isn't resumable
/// must never latch `seeded`, because then every later reconnect re-attaches a replay-less watch and
/// an entry written while NATS was down is never picked up again.
fn is_resumable(cursor: &WatchCursor) -> bool {
    cursor.as_u64().is_some_and(|rev| rev > 0)
}

/// The baseline a scan establishes: the highest KV revision among the scanned entries, which the
/// watch then resumes strictly after. Zero when the scan found nothing — see `is_resumable`.
fn baseline_revision(entries: &[KvEntry]) -> u64 {
    entries
        .iter()
        .filter_map(|e| e.version.as_u64())
        .max()
        .unwrap_or(0)
}

/// Apply a batch of watched deltas to `cur`, returning the set to publish.
///
/// **One clone of the map per batch, not per delta.** The deny-set lives behind an `ArcSwap` and is
/// updated by `rcu`, i.e. clone-on-write: every call here copies the whole O(N) map, so folding a
/// K-delta burst into a single call turns O(K·N) into O(N + K). A single delta — the steady state —
/// is exactly what it always was: one clone.
///
/// `pub` because `benches/unit.rs` measures this seam (batched vs one-at-a-time).
pub fn apply_batch(cur: &DenySet, updates: &[KvUpdate]) -> DenySet {
    let mut set = cur.clone();
    for update in updates {
        match update {
            KvUpdate::Put(e) => {
                if let Some(t) = deny::parse_key(&e.key) {
                    set.insert(t, deny::parse_reason(&e.value));
                }
            }
            // Delete/Purge = restore (explicit delete or TTL expiry).
            KvUpdate::Delete { key, .. } | KvUpdate::Purge { key, .. } => {
                if let Some(t) = deny::parse_key(key) {
                    set.remove(t);
                }
            }
        }
    }
    set
}

/// Build a `DenySet` from KV entries, dropping any whose key isn't a `blackhole.{tenant}`.
fn denyset_from_entries<'a>(entries: impl Iterator<Item = &'a KvEntry>) -> DenySet {
    entries
        .filter_map(|e| Some((deny::parse_key(&e.key)?, deny::parse_reason(&e.value))))
        .collect()
}

/// Rewrite the on-disk snapshot from a fresh scan: truncate, write one `Put` per live entry, and
/// checkpoint the cursor. Returns the reopened writer, or `None` if the rewrite failed (the gateway
/// then runs snapshot-less — the in-memory deny-set is unaffected). Synchronous file I/O, so it runs
/// on a blocking thread off the serving runtime.
async fn rebuild_snapshot(
    path: PathBuf,
    entries: Vec<KvEntry>,
    cursor: WatchCursor,
) -> Option<SnapshotWriter> {
    let res = tokio::task::spawn_blocking(
        move || -> Result<SnapshotWriter, store::snapshot::SnapshotError> {
            // Remove the old log so we don't replay a deleted-but-uncompacted key on a later load.
            // A failed removal is *not* ignorable: if `SnapshotWriter::open` then appends to the
            // surviving file, a compacted-away `Delete` can't undo its stale `Put`, and a later
            // `load()` resurrects a tenant we no longer deny — the exact corruption this rebuild
            // exists to prevent. `NotFound` is the expected, benign case (first boot, or scratch
            // storage); any other error aborts the rebuild so we run snapshot-less rather than on
            // poisoned state.
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
            let mut w = SnapshotWriter::open(&path, SNAPSHOT_COMPACT_THRESHOLD)?;
            // Consume `entries` — it was moved into this closure and `write_update` only borrows,
            // so wrapping each entry in a `Put` by value costs nothing, where cloning it copied a
            // `String` key and a `Vec<u8>` value (2 heap allocations) per scanned entry.
            for e in entries {
                w.write_update(&KvUpdate::Put(e))?;
            }
            w.checkpoint(&cursor)?;
            Ok(w)
        },
    )
    .await;
    match res {
        Ok(Ok(w)) => Some(w),
        Ok(Err(e)) => {
            warn!(error = %e, "snapshot rebuild failed; running without on-disk snapshot");
            None
        }
        Err(e) => {
            warn!(error = %e, "snapshot rebuild task panicked");
            None
        }
    }
}

async fn connect(state: &GatewayState) -> crate::error::Result<Arc<dyn KvStore>> {
    let cfg = &state.config;
    // `expose().to_string()` lifts the creds out of our `Secret` into the plain `String` the store's
    // config requires. This doesn't widen the leak surface: `NatsConnectionConfig` has a hand-written
    // redacting `Debug` (prints `creds: [redacted]`), so a stray `{:?}` on it — in a span, an error
    // context, a reconnect log — can't print the credential. The plaintext copy is necessarily
    // un-zeroized for the connection's life (we hand ownership to the store); same trade-off the pool
    // keys make once they reach Pingora's headers (see `secret`). Redaction, not zeroization, is the
    // control here.
    let conn = NatsConnection::new(NatsConnectionConfig {
        url: cfg.nats_url.clone(),
        creds: cfg.nats_creds.as_ref().map(|s| s.expose().to_string()),
        creds_file: cfg.nats_creds_file.clone(),
    });
    conn.connect().await?;
    let store = conn
        .store_with_config(StoreConfig {
            name: cfg.config_bucket.clone(),
            ..Default::default()
        })
        .await?;
    Ok(store)
}

/// Seed (if needed) and stream one set's deltas until the watch ends or shutdown is signaled.
/// Returns `true` iff it exited because `shutdown` fired — the caller then stops reconnecting.
async fn watch_set<W: WatchedSet>(
    state: &Arc<GatewayState>,
    store: Arc<dyn KvStore>,
    cursor: &mut WatchCursor,
    writer: &mut Option<SnapshotWriter>,
    seeded: &mut bool,
    shutdown: &mut ShutdownWatch,
) -> bool {
    // Seed once, on the first connect that lacks a usable resume point (cold boot with no snapshot,
    // or after a `CursorExpired` reset). A NATS scan is a point-in-time read of the live set; the
    // highest revision among its entries is the baseline the watch resumes strictly after. An empty
    // set ⇒ revision 0, which is *not* a resume point (see `is_resumable`) — we still watch, but we
    // don't claim to be seeded, so the next connect scans again.
    if !*seeded {
        match store.reader().scan(W::PREFIX).await {
            Ok(entries) => {
                let baseline_rev = baseline_revision(&entries);
                let set = W::from_entries(state, entries.iter());
                let count = W::len(&set);
                info!(
                    set = W::NOUN,
                    count,
                    revision = baseline_rev,
                    // Revision 0 means the watch below runs without replay and we'll rescan on the
                    // next connect — worth saying out loud for whoever is reading these lines.
                    resumable = baseline_rev > 0,
                    "seeded from scan"
                );
                W::record_size(state, count);
                W::slot(state).store(Arc::new(set));
                *cursor = WatchCursor::from_u64(baseline_rev);
                // Persist the freshly-scanned baseline so a later restart can skip the scan. We
                // *rebuild* the file (not append): this path runs on a cold boot or after a
                // `CursorExpired` reset, and a stale prior log could otherwise contain a `Put` for a
                // tenant deleted while we were offline — whose `Delete` was compacted away — which a
                // later `load()` would replay and resurrect (wrongly re-denying a tenant). A clean
                // rewrite from the live scan makes the on-disk state exactly match NATS.
                if writer.is_some()
                    && let Some(path) = W::snapshot_path(state)
                {
                    *writer = rebuild_snapshot(PathBuf::from(path), entries, cursor.clone()).await;
                }
                // Latch `seeded` only when the scan produced a cursor we can actually resume from.
                // An empty bucket scans to revision 0, and `watch_prefix_from` degrades that to
                // `DeliverPolicy::New` — no replay. Latching there was the bug: it left the
                // *permanent* state "seeded, resume from 0", so every reconnect for the rest of the
                // process attached a replay-less watch and an entry written while NATS was down was
                // never seen. Leaving it false costs one cheap rescan per reconnect (the bucket is
                // empty, by definition) and closes that hole; the residual gap is the narrow
                // scan→subscribe window of *this* connection, which the next scan re-reads.
                *seeded = is_resumable(cursor);
            }
            Err(e) => {
                // No baseline yet — serve whatever's already in memory (fail-open) and let the
                // reconnect loop retry the scan.
                warn!(set = W::NOUN, error = %e, "scan failed; serving current set, will retry");
                return false;
            }
        }
    }

    // Stream deltas, resuming from the seeded revision so nothing written in the seed→subscribe
    // window is dropped. When the cursor isn't resumable (empty bucket ⇒ revision 0) slipstream
    // degrades this to a bare `watch_prefix` / `DeliverPolicy::New`; `seeded` is false in that case,
    // so the next connect rescans rather than living with a replay-less watch forever.
    let Some(watcher) = store.watcher() else {
        warn!(set = W::NOUN, "store has no watcher; set will not update");
        return false;
    };
    let (tx, mut rx) = tokio::sync::mpsc::channel::<KvUpdate>(WATCH_CHANNEL_CAPACITY);
    let start_cursor = cursor.clone();
    // `watcher` is an owned `Arc` from `store.watcher()` and unused after this, so move it in.
    let watch = tokio::spawn(async move {
        watcher
            .watch_prefix_from(W::PREFIX, &start_cursor, tx)
            .await
    });

    // Deltas are applied in **batches**. `rcu` is clone-on-write over the whole map, so applying a
    // K-delta burst one at a time is O(K·N) copies of an O(N) set — and bursts are the realistic
    // shape here (a control-plane sweep blackholes a batch of tenants at once, and a reconnect
    // replays everything published while we were away). `recv_many` takes whatever is already queued
    // — bounded by the channel's own capacity, so it can't be starved by a fast producer — and the
    // batch lands under one clone, one metric write, and one snapshot checkpoint. The steady state
    // is unchanged: `recv_many` returns as soon as a single update is available, so one delta is
    // still one clone with no added latency.
    //
    // We `select!` on shutdown so a quiet stream (no deltas arriving) doesn't pin the task open
    // through teardown; `select!` can only switch at an await point — between batches — so we never
    // abort mid-`persist_batch`, leaving the snapshot intact.
    let mut batch: Vec<KvUpdate> = Vec::new();
    loop {
        batch.clear();
        let received = tokio::select! {
            _ = shutdown.changed() => {
                watch.abort();
                return true;
            }
            n = rx.recv_many(&mut batch, WATCH_CHANNEL_CAPACITY) => n,
        };
        // Zero means every sender is gone: the watch task has ended.
        if received == 0 {
            break;
        }
        // Capture the new cardinality from inside the closure rather than re-loading the `ArcSwap`
        // afterwards: `rcu` returns the *previous* value, and taking a second guard just to count
        // what we already hold is wasted work. The closure re-runs on contention, and the run that
        // wins is the last one, so this ends up holding the length actually published.
        let mut count = 0usize;
        W::slot(state).rcu(|cur| {
            let set = W::apply_batch(state, cur, &batch);
            count = W::len(&set);
            Arc::new(set)
        });
        W::record_size(state, count);
        // The batch's resume point is its newest delta — that's what the checkpoint below records,
        // and what a reconnect resumes strictly after.
        if let Some(last) = batch.last() {
            *cursor = WatchCursor::from_version(last.version().clone());
        }
        persist_batch(writer, &batch, cursor).await;
    }

    // The watch ended (NATS dropped, or the cursor was compacted away). Inspect why so a compacted
    // cursor forces a fresh scan on the next connect instead of resuming from a dead revision.
    match watch.await {
        Ok(Ok(())) => {}
        Ok(Err(KvError::CursorExpired)) => {
            warn!(
                set = W::NOUN,
                "resume cursor expired (history compacted past it); will rescan"
            );
            *seeded = false;
            *cursor = WatchCursor::none();
        }
        Ok(Err(e)) => warn!(set = W::NOUN, error = %e, "watch ended"),
        Err(e) => warn!(set = W::NOUN, error = %e, "watch task panicked"),
    }
    false
}

/// Append an applied batch to the on-disk snapshot (if configured) and checkpoint the cursor once.
/// `write_update` is a buffered write and cheap; `checkpoint` is the call that flushes (a `write(2)`
/// into the page cache), so checkpointing per batch rather than per delta turns a K-delta burst into
/// one syscall instead of K — and one cursor record is all the batch needs, since it only has to say
/// where the batch left off. `compact` reads+rewrites the whole file, so it's offloaded off the
/// serving runtime when the log crosses its threshold.
///
/// Checkpointing once per batch only ever errs in the safe direction: a crash after some records
/// reached the file but before the cursor record leaves a cursor *older* than the data, so the next
/// boot resumes early and re-applies those deltas — idempotent, since a `Put`/`Delete` of the same
/// key lands the same set. The dangerous direction (a cursor ahead of its data, which would skip
/// deltas) can't happen: the checkpoint is written last.
async fn persist_batch(
    writer: &mut Option<SnapshotWriter>,
    updates: &[KvUpdate],
    cursor: &WatchCursor,
) {
    let needs_compact = match writer.as_mut() {
        Some(w) => {
            for update in updates {
                if let Err(e) = w.write_update(update) {
                    warn!(error = %e, "snapshot write failed");
                }
            }
            match w.checkpoint(cursor) {
                Ok(b) => b,
                Err(e) => {
                    warn!(error = %e, "snapshot checkpoint failed");
                    false
                }
            }
        }
        None => false,
    };
    if needs_compact {
        // Move the writer into a blocking task for the rewrite, then take it back. If it fails we
        // drop the writer (None) and run snapshot-less until the next restart reopens the file —
        // the deny-set itself is unaffected (it lives in the ArcSwap, fed by NATS).
        if let Some(mut w) = writer.take() {
            match tokio::task::spawn_blocking(move || w.compact().map(|()| w)).await {
                Ok(Ok(w)) => *writer = Some(w),
                Ok(Err(e)) => {
                    warn!(error = %e, "snapshot compaction failed; disabling snapshot writer")
                }
                Err(e) => warn!(error = %e, "snapshot compaction task panicked"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deny::DenyReason;
    use store::VersionToken;

    fn entry(key: &str, value: &[u8]) -> KvEntry {
        KvEntry {
            key: key.to_string(),
            value: value.to_vec(),
            version: VersionToken::from_u64(1),
        }
    }

    /// The deny-set's prefix is what the control plane writes to; it used to be a module-level
    /// `const` and now lives in a trait impl, one of several `WatchedSet` implementations. A typo
    /// there fails **silently and completely**: the gateway boots, scans a prefix nothing is written
    /// to, seeds an empty set, and cheerfully serves every denied tenant forever. Nothing else in
    /// this file would notice. Pin the literal.
    #[test]
    fn deny_watches_the_blackhole_prefix() {
        assert_eq!(Deny::PREFIX, "blackhole.");
        // And the keys `deny::parse_key` accepts must actually live under it — the prefix and the
        // parser are two halves of one contract, and agreeing with each other is the whole job.
        assert_eq!(deny::parse_key(&format!("{}42", Deny::PREFIX)), Some(42));
    }

    #[test]
    fn denyset_from_entries_seeds_and_skips_malformed() {
        // This is the seeding core: every boot turns raw KV entries into the live deny-set. A bug
        // here (or a foreign key bleeding through the `filter_map`) means the deny-set is silently
        // wrong at boot — denied tenants served, or unrelated keys denying real tenants.
        let entries = [
            entry("blackhole.42", b"spend"),
            entry("blackhole.99", b"fraud"),
            // Not a `blackhole.{tenant}` key — must be dropped, never inserted as tenant 0 or junk.
            entry("signkey.1", b"spend"),
            // `blackhole.` with a non-numeric tail — `parse_key` rejects it, so it's dropped too.
            entry("blackhole.notanumber", b"spend"),
            // Unrecognized reason value still denies (fail-safe) under `DenyReason::Unknown`.
            entry("blackhole.7", b"mystery"),
        ];

        let set = denyset_from_entries(entries.iter());

        assert_eq!(
            set.len(),
            3,
            "only the three valid blackhole keys are seeded"
        );
        assert_eq!(set.reason(42), Some(DenyReason::Spend));
        assert_eq!(set.reason(99), Some(DenyReason::Fraud));
        assert_eq!(set.reason(7), Some(DenyReason::Unknown));
        // The malformed keys produced no entries (and crucially no spurious tenant 0).
        assert!(!set.is_denied(0));
        assert!(!set.is_denied(1));
    }

    #[test]
    fn denyset_from_entries_empty_is_allow_all() {
        let set = denyset_from_entries([].iter());
        assert!(set.is_empty());
        assert!(!set.is_denied(42)); // default-allow on a cold/empty scan
    }

    #[test]
    fn apply_batch_matches_one_at_a_time() {
        // Batching is a performance change only: a batch must land exactly the set that applying
        // its deltas one `rcu` at a time would have. Ordering inside the batch is what makes this
        // non-trivial — a Put followed by a Delete of the same tenant must end deleted, and the
        // reverse must end denied.
        let updates = vec![
            KvUpdate::Put(entry("blackhole.1", b"spend")),
            KvUpdate::Put(entry("blackhole.2", b"fraud")),
            // Re-Put of a live tenant: the newer reason wins.
            KvUpdate::Put(entry("blackhole.1", b"fraud")),
            // Delete after Put (TTL expiry on a spend hold) → restored.
            KvUpdate::Delete {
                key: "blackhole.2".to_string(),
                version: VersionToken::from_u64(4),
            },
            // Purge of a tenant that was never denied → no-op, not an insert.
            KvUpdate::Purge {
                key: "blackhole.9".to_string(),
                version: VersionToken::from_u64(5),
            },
            // Foreign key in the stream must not corrupt the set.
            KvUpdate::Put(entry("signkey.1", b"spend")),
        ];

        let batched = apply_batch(&DenySet::new(), &updates);
        let mut sequential = DenySet::new();
        for u in &updates {
            sequential = apply_batch(&sequential, std::slice::from_ref(u));
        }

        assert_eq!(batched.len(), sequential.len());
        assert_eq!(batched.len(), 1, "only tenant 1 survives the batch");
        assert_eq!(batched.reason(1), Some(DenyReason::Fraud)); // last Put wins
        assert_eq!(sequential.reason(1), Some(DenyReason::Fraud));
        assert!(!batched.is_denied(2)); // deleted
        assert!(!batched.is_denied(9)); // purge of an absent tenant
        assert!(!batched.is_denied(0)); // `signkey.1` never became tenant 0
    }

    #[test]
    fn revision_zero_is_not_a_resume_point() {
        // slipstream resumes a watch only when the cursor's revision is > 0 (see
        // `NatsKvWatcher::watch_prefix_from`: `Some(rev) if rev > 0`, else it delegates to
        // `watch_prefix` = `DeliverPolicy::New`). A revision-0 cursor is therefore *present but
        // useless*, which is why `!cursor.is_none()` is the wrong test everywhere in this file.
        assert!(!is_resumable(&WatchCursor::none()));
        assert!(!is_resumable(&WatchCursor::from_u64(0)));
        assert!(!WatchCursor::from_u64(0).is_none()); // ...and it does *not* look absent
        assert!(is_resumable(&WatchCursor::from_u64(1)));
        assert!(is_resumable(&WatchCursor::from_u64(u64::MAX)));
    }

    #[test]
    fn empty_scan_yields_no_resume_point_so_we_rescan() {
        // The seed decision in `watch_deny`: baseline = highest revision in the scan, and `seeded`
        // is latched only if that baseline is resumable. An empty bucket ⇒ 0 ⇒ not seeded ⇒ the
        // next connect scans again. Latching here was the bug: it pinned every future reconnect to
        // a replay-less watch, so a `blackhole.X` written while NATS was down was never applied for
        // the life of the process.
        let empty = baseline_revision(&[]);
        assert_eq!(empty, 0);
        assert!(!is_resumable(&WatchCursor::from_u64(empty)));

        // A non-empty scan does establish one — the highest revision seen, not the first or last.
        let mut lo = entry("blackhole.1", b"spend");
        lo.version = VersionToken::from_u64(7);
        let mut hi = entry("blackhole.2", b"fraud");
        hi.version = VersionToken::from_u64(9);
        let scanned = baseline_revision(&[hi, lo]);
        assert_eq!(scanned, 9);
        assert!(is_resumable(&WatchCursor::from_u64(scanned)));
    }

    #[test]
    fn backoff_doubles_to_the_cap() {
        let mut b = ReconnectBackoff::new();
        assert_eq!(b.delay(), RECONNECT_BACKOFF_BASE);
        for expected in [2, 4, 8, 16, 30, 30] {
            b.grow();
            assert_eq!(b.delay(), Duration::from_secs(expected));
        }
    }

    #[test]
    fn backoff_grows_when_only_the_watch_keeps_failing() {
        // The regression: NATS accepts the connection every time but the watch dies immediately
        // (bucket missing, scan permission denied, no watcher, consumer-create refused). The old
        // code reset the backoff on connect, so the doubling below was undone every iteration and
        // the loop reconnected at the 1s base forever — ~86k connects and 86k warn lines a day.
        let mut b = ReconnectBackoff::new();
        for _ in 0..10 {
            b.credit(Duration::from_millis(20)); // watch died right after connecting
            b.grow();
        }
        assert_eq!(
            b.delay(),
            RECONNECT_BACKOFF_MAX,
            "a connect that never yields a working watch must still back off"
        );
    }

    #[test]
    fn backoff_resets_only_after_a_productive_watch() {
        let mut b = ReconnectBackoff::new();
        for _ in 0..10 {
            b.grow();
        }
        assert_eq!(b.delay(), RECONNECT_BACKOFF_MAX);

        // Just short of the cap is still flapping — keep backing off.
        b.credit(RECONNECT_BACKOFF_MAX - Duration::from_millis(1));
        assert_eq!(b.delay(), RECONNECT_BACKOFF_MAX);

        // A watch that outlived the cap was doing real work: recovery is immediate again.
        b.credit(RECONNECT_BACKOFF_MAX);
        assert_eq!(b.delay(), RECONNECT_BACKOFF_BASE);
    }
}
