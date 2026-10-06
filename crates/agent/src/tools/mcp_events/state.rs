//! A session's durable MCP Events state: each subscription's cursor and dedup window, the events
//! accepted from servers but not yet delivered to the model (`pending`), and the server-identity
//! keys seen per origin. One file beside the transcript (`<session>.mcp-events.json`).
//!
//! **Writes.** One writer task owns the file. It writes the way the session store writes — a
//! private temp file, `fsync`, `rename`, then the directory `fsync` — so a reader or a restart sees
//! the old file or the new one, never a torn one. Every write is a snapshot taken under the lock and
//! writes happen strictly one after another, so the newest snapshot is always the last one on disk.
//! Ordinary changes are coalesced (one write per 100 ms at most); [`StateStore::commit`] waits for a
//! write that includes everything up to the call — that is what a webhook `2xx` and an injection
//! into the model wait on. Reads and writes run on the blocking pool.
//!
//! **Lifetime.** When the last handle goes (the hub dropped, with or without a clean shutdown) the
//! writer writes once more and ends. [`StateStore::relocate`] follows the session to a new file
//! (`new_session`, `switch_session`).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;
use tokio::sync::{Notify, mpsc, oneshot, watch};

use crate::settings::McpEventAction;

use super::SPEC_COMMIT;

/// One subscription's persisted position: the last safe cursor and the newest event ids seen.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub(super) struct PersistedSub {
    pub(super) cursor: Option<String>,
    #[serde(default)]
    pub(super) recent: Vec<String>,
}

/// An event a server has been told was received (its cursor advanced, its webhook acked) that the
/// model has not yet seen. It stays here — on disk — until the prompt carrying it has run and the
/// transcript holding it has been persisted; a restart re-injects whatever is left.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(super) struct PendingEvent {
    pub(super) seq: u64,
    pub(super) action: McpEventAction,
    pub(super) server: String,
    pub(super) name: String,
    pub(super) arguments: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) instructions: Option<String>,
    /// The occurrence, or for a gap notice `{"gap": true, "cursor": …}`.
    pub(super) event: Value,
    /// The injection batch carrying it right now. Not persisted: after a restart every pending
    /// event is ready again.
    #[serde(skip)]
    pub(super) batch: Option<u64>,
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
struct PersistedFile {
    #[serde(default)]
    spec_commit: String,
    #[serde(default)]
    subscriptions: BTreeMap<String, PersistedSub>,
    #[serde(default)]
    pending: Vec<PendingEvent>,
    /// Origin → base64 Ed25519 public keys ever published there (never shrinks; see
    /// `webhook::KEY_POLICY`).
    #[serde(default)]
    server_keys: BTreeMap<String, Vec<String>>,
}

#[derive(Default)]
struct Data {
    path: Option<PathBuf>,
    subs: BTreeMap<String, PersistedSub>,
    pending: Vec<PendingEvent>,
    next_seq: u64,
    server_keys: BTreeMap<String, Vec<String>>,
}

enum Req {
    Dirty,
    Commit(oneshot::Sender<()>),
}

struct Inner {
    data: Arc<Mutex<Data>>,
    tx: mpsc::UnboundedSender<Req>,
    loaded: watch::Receiver<bool>,
    changed: Arc<Notify>,
    max_pending: usize,
    #[cfg_attr(not(test), allow(dead_code))]
    writer: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

/// See the module docs.
#[derive(Clone)]
pub(super) struct StateStore {
    inner: Arc<Inner>,
}

fn lock(data: &Mutex<Data>) -> std::sync::MutexGuard<'_, Data> {
    data.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl StateStore {
    /// Open (or create, on first write) the state at `path`; `None` keeps everything in memory —
    /// pending events still queue, nothing survives the process.
    pub(super) fn open(path: Option<PathBuf>, max_pending: usize) -> Self {
        let data = Arc::new(Mutex::new(Data {
            path: path.clone(),
            next_seq: 1,
            ..Data::default()
        }));
        let (tx, rx) = mpsc::unbounded_channel();
        let (loaded_tx, loaded) = watch::channel(false);
        let changed = Arc::new(Notify::new());
        let writer = tokio::spawn(writer(data.clone(), rx, loaded_tx, path, changed.clone()));
        Self {
            inner: Arc::new(Inner {
                data,
                tx,
                loaded,
                changed,
                max_pending,
                writer: Mutex::new(Some(writer)),
            }),
        }
    }

    /// Wait until the file on disk (if any) has been read.
    pub(super) async fn loaded(&self) {
        let mut rx = self.inner.loaded.clone();
        let _ = rx.wait_for(|l| *l).await;
    }

    /// Notified whenever pending events are added or released.
    pub(super) fn changed(&self) -> Arc<Notify> {
        self.inner.changed.clone()
    }

    fn dirty(&self) {
        let _ = self.inner.tx.send(Req::Dirty);
    }

    pub(super) fn sub(&self, key: &str) -> Option<PersistedSub> {
        lock(&self.inner.data).subs.get(key).cloned()
    }

    pub(super) fn set_sub(&self, key: &str, entry: PersistedSub) {
        {
            let mut d = lock(&self.inner.data);
            if d.subs.get(key) == Some(&entry) {
                return;
            }
            d.subs.insert(key.to_owned(), entry);
        }
        self.dirty();
    }

    /// An explicit unsubscribe forgets the position; a session ending keeps it, to resume from.
    pub(super) fn forget_sub(&self, key: &str) {
        if lock(&self.inner.data).subs.remove(key).is_some() {
            self.dirty();
        }
    }

    /// Queue an event for the model. `false` when [`Self::open`]'s `max_pending` are already
    /// waiting: the caller must then *not* acknowledge or advance past it (backpressure).
    pub(super) fn push_pending(&self, mut event: PendingEvent) -> bool {
        {
            let mut d = lock(&self.inner.data);
            if d.pending.len() >= self.inner.max_pending {
                return false;
            }
            event.seq = d.next_seq;
            event.batch = None;
            d.next_seq += 1;
            d.pending.push(event);
        }
        self.dirty();
        self.inner.changed.notify_one();
        true
    }

    pub(super) fn pending_len(&self) -> usize {
        lock(&self.inner.data).pending.len()
    }

    /// Pending events not yet carried by an injection, oldest first.
    pub(super) fn ready_pending(&self) -> Vec<PendingEvent> {
        lock(&self.inner.data)
            .pending
            .iter()
            .filter(|p| p.batch.is_none())
            .cloned()
            .collect()
    }

    pub(super) fn assign_batch(&self, seqs: &[u64], batch: u64) {
        let mut d = lock(&self.inner.data);
        for p in d.pending.iter_mut().filter(|p| seqs.contains(&p.seq)) {
            p.batch = Some(batch);
        }
    }

    /// Put a batch's events back in the ready queue (its injection never reached the session).
    pub(super) fn unassign(&self, batch: u64) {
        let mut d = lock(&self.inner.data);
        for p in d.pending.iter_mut().filter(|p| p.batch == Some(batch)) {
            p.batch = None;
        }
        drop(d);
        self.inner.changed.notify_one();
    }

    /// The model has seen `batch` (its prompt ran and the transcript holding it is persisted).
    pub(super) fn delivered(&self, batch: u64) -> bool {
        let removed = {
            let mut d = lock(&self.inner.data);
            let before = d.pending.len();
            d.pending.retain(|p| p.batch != Some(batch));
            before != d.pending.len()
        };
        if removed {
            self.dirty();
            self.inner.changed.notify_one();
        }
        removed
    }

    /// Every server-identity key ever seen for `origin`, base64.
    pub(super) fn keys(&self, origin: &str) -> Vec<String> {
        lock(&self.inner.data)
            .server_keys
            .get(origin)
            .cloned()
            .unwrap_or_default()
    }

    /// Record keys for `origin`; the set only grows. `true` if anything was new.
    pub(super) fn add_keys(&self, origin: &str, keys: &[String]) -> bool {
        let added = {
            let mut d = lock(&self.inner.data);
            let set = d.server_keys.entry(origin.to_owned()).or_default();
            let mut added = false;
            for k in keys {
                if !set.contains(k) {
                    set.push(k.clone());
                    added = true;
                }
            }
            added
        };
        if added {
            self.dirty();
        }
        added
    }

    /// Follow the session to a new transcript file: the state is written there from now on.
    pub(super) fn relocate(&self, path: Option<PathBuf>) {
        {
            let mut d = lock(&self.inner.data);
            if d.path == path {
                return;
            }
            d.path = path;
        }
        self.dirty();
    }

    /// Wait for a write that includes every change made before this call. Returns at once when the
    /// writer is gone (nothing more can be made durable).
    pub(super) async fn commit(&self) {
        let (tx, rx) = oneshot::channel();
        if self.inner.tx.send(Req::Commit(tx)).is_ok() {
            let _ = tokio::time::timeout(Duration::from_secs(10), rx).await;
        }
    }

    /// The writer's handle, for tests that need to see it end.
    #[cfg(test)]
    fn take_writer(&self) -> Option<tokio::task::JoinHandle<()>> {
        self.inner
            .writer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }
}

async fn writer(
    data: Arc<Mutex<Data>>,
    mut rx: mpsc::UnboundedReceiver<Req>,
    loaded: watch::Sender<bool>,
    path: Option<PathBuf>,
    changed: Arc<Notify>,
) {
    if let Some(path) = path {
        let read = tokio::task::spawn_blocking(move || std::fs::read(path)).await;
        if let Ok(Ok(bytes)) = read
            && let Ok(file) = serde_json::from_slice::<PersistedFile>(&bytes)
        {
            let mut d = lock(&data);
            d.next_seq = file.pending.iter().map(|p| p.seq).max().unwrap_or(0) + 1;
            d.subs = file.subscriptions;
            d.pending = file.pending;
            d.server_keys = file.server_keys;
        }
    }
    let _ = loaded.send(true);
    changed.notify_one();
    loop {
        let Some(first) = rx.recv().await else {
            // Every handle is gone: one last write, then end.
            write_snapshot(&data).await;
            return;
        };
        let mut commits = Vec::new();
        match first {
            Req::Commit(tx) => commits.push(tx),
            Req::Dirty => tokio::time::sleep(Duration::from_millis(100)).await,
        }
        let mut closed = false;
        loop {
            match rx.try_recv() {
                Ok(Req::Commit(tx)) => commits.push(tx),
                Ok(Req::Dirty) => {}
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    closed = true;
                    break;
                }
            }
        }
        write_snapshot(&data).await;
        for tx in commits {
            let _ = tx.send(());
        }
        if closed {
            return;
        }
    }
}

async fn write_snapshot(data: &Arc<Mutex<Data>>) {
    let (path, bytes) = {
        let d = lock(data);
        let Some(path) = d.path.clone() else { return };
        let file = PersistedFile {
            spec_commit: SPEC_COMMIT.to_owned(),
            subscriptions: d.subs.clone(),
            pending: d.pending.clone(),
            server_keys: d.server_keys.clone(),
        };
        let Ok(bytes) = serde_json::to_vec(&file) else {
            return;
        };
        (path, bytes)
    };
    let result = tokio::task::spawn_blocking(move || write_atomic(&path, &bytes)).await;
    if let Ok(Err(e)) = result {
        tracing::warn!(error = %e, "could not persist MCP Events state");
    }
}

fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;
    let tmp = path.with_extension(format!("json.{}.tmp", crate::tools::temp_suffix()));
    let write = (|| -> std::io::Result<()> {
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            opts.mode(0o600);
        }
        let mut f = opts.open(&tmp)?;
        f.write_all(bytes)?;
        f.flush()?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&tmp, path)?;
        if let Some(dir) = path.parent() {
            std::fs::File::open(dir)?.sync_all()?;
        }
        Ok(())
    })();
    if write.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    write
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn event(n: u64) -> PendingEvent {
        PendingEvent {
            seq: 0,
            action: McpEventAction::FollowUp,
            server: "s".into(),
            name: "n".into(),
            arguments: serde_json::json!({}),
            instructions: None,
            event: serde_json::json!({ "eventId": format!("e{n}") }),
            batch: None,
        }
    }

    fn on_disk(path: &std::path::Path) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }

    #[tokio::test]
    async fn dropping_every_handle_writes_once_more_and_ends_the_writer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.mcp-events.json");
        let store = StateStore::open(Some(path.clone()), 10);
        store.loaded().await;
        store.set_sub(
            "k",
            PersistedSub {
                cursor: Some("7".into()),
                recent: vec!["a".into()],
            },
        );
        let writer = store.take_writer().unwrap();
        drop(store); // no commit, no shutdown
        tokio::time::timeout(Duration::from_secs(2), writer)
            .await
            .expect("the writer ends once its last handle is gone")
            .unwrap();
        assert_eq!(on_disk(&path)["subscriptions"]["k"]["cursor"], "7");
    }

    #[tokio::test]
    async fn the_newest_snapshot_always_wins() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.mcp-events.json");
        let store = StateStore::open(Some(path.clone()), 10);
        store.loaded().await;
        for i in 0..200 {
            store.set_sub(
                "k",
                PersistedSub {
                    cursor: Some(i.to_string()),
                    recent: vec![],
                },
            );
            if i % 37 == 0 {
                store.commit().await;
            }
        }
        store.commit().await;
        assert_eq!(on_disk(&path)["subscriptions"]["k"]["cursor"], "199");
    }

    #[tokio::test]
    async fn pending_events_survive_a_reopen_ready_again_and_backpressure_at_the_cap() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.mcp-events.json");
        let store = StateStore::open(Some(path.clone()), 2);
        store.loaded().await;
        assert!(store.push_pending(event(1)));
        assert!(store.push_pending(event(2)));
        assert!(
            !store.push_pending(event(3)),
            "at the cap the event is refused, not dropped"
        );
        let ready = store.ready_pending();
        store.assign_batch(&[ready[0].seq], 9);
        assert_eq!(store.ready_pending().len(), 1);
        store.commit().await;
        drop(store);

        let again = StateStore::open(Some(path), 2);
        again.loaded().await;
        let ready = again.ready_pending();
        assert_eq!(
            ready.len(),
            2,
            "an in-flight batch is ready again after a restart"
        );
        assert_eq!(ready[0].event["eventId"], "e1");
        again.assign_batch(&[ready[0].seq, ready[1].seq], 1);
        assert!(again.delivered(1));
        assert_eq!(again.pending_len(), 0);
    }

    #[tokio::test]
    async fn relocating_writes_to_the_new_file_and_keys_only_grow() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.mcp-events.json");
        let b = dir.path().join("b.mcp-events.json");
        let store = StateStore::open(Some(a.clone()), 2);
        store.loaded().await;
        assert!(store.add_keys("https://x", &["k1".into()]));
        assert!(!store.add_keys("https://x", &["k1".into()]));
        store.commit().await;
        store.relocate(Some(b.clone()));
        store.set_sub("k", PersistedSub::default());
        store.commit().await;
        assert_eq!(on_disk(&b)["server_keys"]["https://x"][0], "k1");
        assert!(on_disk(&b)["subscriptions"]["k"].is_object());
    }
}
