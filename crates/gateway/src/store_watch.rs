//! slipstream deny-set watcher — the gateway's **only** use of NATS.
//!
//! Seeds the deny-set at boot, then streams deltas. **Fail-open**: a NATS blip keeps the last-known
//! set (we never clear), so an outage degrades to a stale deny-set, not "reject everything". Auth
//! and pool/signing keys come from config, so they're unaffected by NATS being down — only
//! spend/fraud enforcement goes stale.
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
//! Either way the watch resumes from a **revision** (`watch_prefix_from`), never a bare
//! `watch_prefix`: the latter uses NATS `DeliverPolicy::New` (no replay), so a deny entry written in
//! the window between seeding and the subscription attaching would be silently lost. Resuming from
//! the seeded revision closes that window with no gap and no double-apply (it starts strictly after
//! the seeded revision). If the backend compacted past the cursor (`CursorExpired`), we drop back to
//! a fresh scan, which re-establishes a valid baseline.
//!
//! Runs as a Pingora `BackgroundService` so the NATS client is created on the serving runtime
//! (async-nats ties its tasks to the runtime it's built on; connecting earlier would break it).

use crate::deny::{self, DenySet};
use crate::state::GatewayState;
use async_trait::async_trait;
use pingora_core::server::ShutdownWatch;
use pingora_core::services::background::BackgroundService;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use store::snapshot::SnapshotWriter;
use store::{
    Connection, KvEntry, KvError, KvStore, KvUpdate, NatsConnection, NatsConnectionConfig,
    StoreConfig, WatchCursor,
};
use tracing::{error, info, warn};

const BLACKHOLE_PREFIX: &str = "blackhole.";

/// Compact the on-disk snapshot once it grows past this many bytes of appended deltas. The deny-set
/// is low-churn, so this is rarely hit; it just bounds the log if a tenant flaps.
const SNAPSHOT_COMPACT_THRESHOLD: u64 = 1024 * 1024;

/// Reconnect backoff bounds: start at 1s, double to a 30s ceiling. Generous enough to stop log
/// spam during a long NATS outage, tight enough that recovery is near-immediate once it returns.
const RECONNECT_BACKOFF_BASE: Duration = Duration::from_secs(1);
const RECONNECT_BACKOFF_MAX: Duration = Duration::from_secs(30);

pub struct WatcherService {
    pub state: Arc<GatewayState>,
}

#[async_trait]
impl BackgroundService for WatcherService {
    async fn start(&self, mut shutdown: ShutdownWatch) {
        // Resume position + on-disk snapshot writer persist across reconnects: a NATS blip resumes
        // the watch from `cursor` instead of re-scanning, and `seeded` stays true so we don't reseed.
        let mut cursor = WatchCursor::none();
        let mut writer: Option<SnapshotWriter> = None;
        let mut seeded = false;

        if let Some(path) = self.state.config.snapshot_path.clone() {
            let path = PathBuf::from(path);
            // Snapshot I/O is synchronous (whole-file read/rewrite) — offload it so we never stall
            // the serving runtime this BackgroundService shares with the proxy.
            let load_path = path.clone();
            match tokio::task::spawn_blocking(move || store::snapshot::load(&load_path)).await {
                Ok(Ok(Some(snap))) => {
                    let set = denyset_from_entries(snap.entries.values());
                    info!(count = set.len(), "seeded deny-set from on-disk snapshot");
                    self.state.metrics.deny_set_size.set(set.len() as i64);
                    self.state.deny.store(Arc::new(set));
                    // A snapshot without a saved cursor can't safely resume (a bare watch would
                    // race), so only treat it as seeded when it carries a resume point; otherwise
                    // fall through to a NATS scan on connect.
                    if !snap.cursor.is_none() {
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

        // Reconnect backoff: 1s doubling to a 30s cap, reset on every successful connect. A fixed
        // 2s retry hammered the log at a constant rate through a long outage (minutes to hours),
        // burying other signals during the very incident an oncall is reading these logs for. The
        // gateway serves correctly on the stale set throughout — this is purely about log volume
        // and not pointlessly spinning on a down NATS.
        let mut backoff = RECONNECT_BACKOFF_BASE;
        loop {
            // Connect, but bail immediately if Pingora signals shutdown mid-connect (e.g. NATS is
            // down and `connect` is retrying its own backoff) rather than blocking teardown.
            let store = tokio::select! {
                _ = shutdown.changed() => {
                    info!("shutdown signaled; deny-set watcher exiting");
                    return;
                }
                outcome = connect(&self.state) => match outcome {
                    Ok(store) => store,
                    Err(e) => {
                        self.state.metrics.nats_connected.set(0);
                        error!(error = %e, backoff_secs = backoff.as_secs(), "slipstream connect failed; retrying");
                        // Reconnect backoff, also interruptible by shutdown.
                        tokio::select! {
                            _ = shutdown.changed() => return,
                            _ = tokio::time::sleep(backoff) => {
                                backoff = (backoff * 2).min(RECONNECT_BACKOFF_MAX);
                                continue;
                            }
                        }
                    }
                },
            };

            backoff = RECONNECT_BACKOFF_BASE;
            self.state.metrics.nats_connected.set(1);
            info!("slipstream connected; watching deny-set");
            // `watch_deny` returns `true` when it exited because shutdown was signaled — stop the
            // reconnect loop cleanly instead of trying to reconnect a shutting-down process.
            if watch_deny(
                &self.state,
                store,
                &mut cursor,
                &mut writer,
                &mut seeded,
                &mut shutdown,
            )
            .await
            {
                info!("shutdown signaled; deny-set watcher exiting");
                return;
            }
            self.state.metrics.nats_connected.set(0);
            warn!("deny-set watch exited; reconnecting");
            tokio::select! {
                _ = shutdown.changed() => return,
                _ = tokio::time::sleep(backoff) => {
                    backoff = (backoff * 2).min(RECONNECT_BACKOFF_MAX);
                }
            }
        }
    }
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
            for e in &entries {
                w.write_update(&KvUpdate::Put(e.clone()))?;
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

/// Seed (if needed) and stream deny-set deltas until the watch ends or shutdown is signaled.
/// Returns `true` iff it exited because `shutdown` fired — the caller then stops reconnecting.
async fn watch_deny(
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
    // set ⇒ revision 0 ⇒ resume from the start of history (the deny bucket is low-churn, so a full
    // replay is cheap and still gap-free).
    if !*seeded {
        match store.reader().scan(BLACKHOLE_PREFIX).await {
            Ok(entries) => {
                let baseline_rev = entries
                    .iter()
                    .filter_map(|e| e.version.as_u64())
                    .max()
                    .unwrap_or(0);
                let set = denyset_from_entries(entries.iter());
                info!(
                    count = set.len(),
                    revision = baseline_rev,
                    "seeded deny-set from scan"
                );
                state.metrics.deny_set_size.set(set.len() as i64);
                state.deny.store(Arc::new(set));
                *cursor = WatchCursor::from_u64(baseline_rev);
                // Persist the freshly-scanned baseline so a later restart can skip the scan. We
                // *rebuild* the file (not append): this path runs on a cold boot or after a
                // `CursorExpired` reset, and a stale prior log could otherwise contain a `Put` for a
                // tenant deleted while we were offline — whose `Delete` was compacted away — which a
                // later `load()` would replay and resurrect (wrongly re-denying a tenant). A clean
                // rewrite from the live scan makes the on-disk state exactly match NATS.
                if writer.is_some() {
                    if let Some(path) = state.config.snapshot_path.clone() {
                        *writer =
                            rebuild_snapshot(PathBuf::from(path), entries, cursor.clone()).await;
                    }
                }
                *seeded = true;
            }
            Err(e) => {
                // No baseline yet — serve whatever's already in memory (fail-open) and let the
                // reconnect loop retry the scan.
                warn!(error = %e, "deny-set scan failed; serving current set, will retry");
                return false;
            }
        }
    }

    // Stream deltas, resuming from the seeded revision. Never a bare `watch_prefix` (DeliverPolicy::
    // New) — that would drop anything written in the seed→subscribe window.
    let Some(watcher) = store.watcher() else {
        warn!("store has no watcher; deny-set will not update");
        return false;
    };
    let (tx, mut rx) = tokio::sync::mpsc::channel::<KvUpdate>(256);
    let w = watcher.clone();
    let start_cursor = cursor.clone();
    let watch = tokio::spawn(async move {
        w.watch_prefix_from(BLACKHOLE_PREFIX, &start_cursor, tx)
            .await
    });

    // Updates are rcu (clone-on-write); the set is tiny (O(denied)). Each applied delta also
    // advances the in-memory cursor (so a reconnect resumes from here) and is appended to the
    // on-disk snapshot if one is configured. We `select!` on shutdown so a quiet stream (no deltas
    // arriving) doesn't pin the task open through teardown; `select!` can only switch at an await
    // point — between updates — so we never abort mid-`persist_update`, leaving the snapshot intact.
    loop {
        let update = tokio::select! {
            _ = shutdown.changed() => {
                watch.abort();
                return true;
            }
            update = rx.recv() => match update {
                Some(u) => u,
                None => break,
            },
        };
        state.deny.rcu(|cur| {
            let mut set = (**cur).clone();
            match &update {
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
            Arc::new(set)
        });
        // Reflect the new cardinality. A lock-free load of the set we just swapped in — cheap, and
        // the deltas are low-churn, so this is far off any hot path.
        state
            .metrics
            .deny_set_size
            .set(state.deny.load().len() as i64);
        *cursor = WatchCursor::from_version(update.version().clone());
        persist_update(writer, &update, cursor).await;
    }

    // The watch ended (NATS dropped, or the cursor was compacted away). Inspect why so a compacted
    // cursor forces a fresh scan on the next connect instead of resuming from a dead revision.
    match watch.await {
        Ok(Ok(())) => {}
        Ok(Err(KvError::CursorExpired)) => {
            warn!("deny-set resume cursor expired (history compacted past it); will rescan");
            *seeded = false;
            *cursor = WatchCursor::none();
        }
        Ok(Err(e)) => warn!(error = %e, "deny-set watch ended"),
        Err(e) => warn!(error = %e, "deny-set watch task panicked"),
    }
    false
}

/// Append one applied delta to the on-disk snapshot (if configured) and checkpoint the cursor.
/// `write_update`/`checkpoint` are buffered/`write(2)` and cheap; `compact` reads+rewrites the whole
/// file, so it's offloaded off the serving runtime when the log crosses its threshold.
async fn persist_update(
    writer: &mut Option<SnapshotWriter>,
    update: &KvUpdate,
    cursor: &WatchCursor,
) {
    let needs_compact = match writer.as_mut() {
        Some(w) => {
            if let Err(e) = w.write_update(update) {
                warn!(error = %e, "snapshot write failed");
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
}
