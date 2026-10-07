//! A session's durable MCP Events state, in two files beside the transcript:
//!
//! - `<session>.mcp-events.json` — the **snapshot**: each subscription's cursor, dedup window and
//!   webhook callback (token and secret), and the server-identity keys seen per origin. Small and
//!   bounded (it never holds events), rewritten whole when it changes: a private temp file, `fsync`,
//!   `rename`, then the directory `fsync`, so a reader sees the old file or the new one.
//! - `<session>.mcp-events.log` — the **pending log**: the events accepted from servers but not yet
//!   delivered to the model, as appended JSON lines (`{"add": …}` / `{"done": [seq, …]}`). A commit
//!   appends only what changed since the last one and `fdatasync`s it, so its cost is the size of the
//!   change, not of the queue. When dead records outweigh live ones the log is rewritten (compacted)
//!   with only the live events — amortized, never on the hot path's critical size. A torn last line
//!   (a crash mid-append) is ignored on replay.
//!
//! **Bounds.** The queue is bounded in count *and* in bytes ([`StateStore::open`]); past either, an
//! event is refused (`push_pending` is `false`) and the caller applies backpressure.
//!
//! **Writes.** One writer task owns both files; writes happen strictly one after another, so the
//! newest state is always the last on disk. Ordinary changes are coalesced (one write per 100 ms at
//! most); [`StateStore::commit`] waits for a write that includes everything up to the call and says
//! whether it succeeded — that is what a webhook `2xx`, a poll cursor advance and an injection into
//! the model wait on. The log is written before the snapshot, and the snapshot is not written when
//! the log could not be, so a persisted cursor never runs ahead of the events it passed. A failed
//! write is retried by the next one; nothing is dropped. Nothing is written while there is nothing
//! to persist: a session that never subscribes leaves no files.
//!
//! **Lifetime.** When the last handle goes the writer writes once more and ends.
//! [`StateStore::relocate`] moves the state to a new transcript's files (`new_session`,
//! `switch_session`), merging whatever that transcript already had.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;
use tokio::sync::{Notify, mpsc, oneshot, watch};

use crate::settings::McpEventAction;

use super::SPEC_COMMIT;

/// How long a commit waits for the writer before reporting failure.
const COMMIT_TIMEOUT: Duration = Duration::from_secs(10);
/// The log is compacted once it is larger than this and more than twice its live content.
const COMPACT_MIN_BYTES: u64 = 1024 * 1024;

/// A webhook subscription's callback, kept across restarts so deliveries the server retries while
/// this process is down still land on a route and verify.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(super) struct PersistedWebhook {
    pub(super) token: String,
    /// The `whsec_…` secret the server was last confirmed to sign with.
    pub(super) secret: String,
}

/// One subscription's persisted position: the last safe cursor and the newest event ids seen.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub(super) struct PersistedSub {
    pub(super) cursor: Option<String>,
    #[serde(default)]
    pub(super) recent: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) webhook: Option<PersistedWebhook>,
}

/// An event a server has been told was received (its cursor advanced, its webhook acked) that the
/// model has not yet seen. It stays in the log until the prompt carrying it has reached the model
/// and the transcript holding it has been persisted; a restart re-injects whatever is left.
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
    /// Its size in the log, for the byte bound and compaction.
    #[serde(skip)]
    size: u64,
    /// Injections of it that did not reach the model (this process only).
    #[serde(skip)]
    attempts: u32,
}

impl PendingEvent {
    pub(super) fn new(
        action: McpEventAction,
        server: String,
        name: String,
        arguments: Value,
        instructions: Option<String>,
        event: Value,
    ) -> Self {
        PendingEvent {
            seq: 0,
            action,
            server,
            name,
            arguments,
            instructions,
            event,
            batch: None,
            size: 0,
            attempts: 0,
        }
    }
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
struct Snapshot {
    #[serde(default)]
    spec_commit: String,
    #[serde(default)]
    subscriptions: BTreeMap<String, PersistedSub>,
    /// Origin → base64 Ed25519 public keys ever published there (never shrinks; see the key policy
    /// in `webhook`).
    #[serde(default)]
    server_keys: BTreeMap<String, Vec<String>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum LogOp {
    Add(PendingEvent),
    Done(Vec<u64>),
}

fn log_line(op: &LogOp) -> Vec<u8> {
    let mut line = serde_json::to_vec(op).unwrap_or_default();
    line.push(b'\n');
    line
}

/// The pending log's path for a snapshot path.
fn log_path(snapshot: &Path) -> PathBuf {
    snapshot.with_extension("log")
}

#[derive(Default)]
struct Data {
    path: Option<PathBuf>,
    subs: BTreeMap<String, PersistedSub>,
    server_keys: BTreeMap<String, Vec<String>>,
    pending: Vec<PendingEvent>,
    pending_bytes: u64,
    next_seq: u64,
    /// Log records not yet appended, already serialized.
    unwritten: Vec<u8>,
    snapshot_dirty: bool,
    /// Bytes in the log file on disk.
    log_len: u64,
}

impl Data {
    fn is_empty(&self) -> bool {
        self.subs.is_empty() && self.server_keys.is_empty() && self.pending.is_empty()
    }
}

enum Req {
    Dirty,
    Commit(oneshot::Sender<Result<(), String>>),
    Relocate(Option<PathBuf>, oneshot::Sender<()>),
}

struct Inner {
    data: Arc<Mutex<Data>>,
    tx: mpsc::UnboundedSender<Req>,
    loaded: watch::Receiver<bool>,
    changed: Arc<Notify>,
    max_pending: usize,
    max_bytes: u64,
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
    /// Open the state whose snapshot is `path` (files are created on the first write that has
    /// something to persist); `None` keeps everything in memory — pending events still queue,
    /// nothing survives the process. At most `max_pending` events and `max_bytes` of them wait.
    pub(super) fn open(path: Option<PathBuf>, max_pending: usize, max_bytes: u64) -> Self {
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
                max_bytes,
                writer: Mutex::new(Some(writer)),
            }),
        }
    }

    /// Wait until the files on disk (if any) have been read.
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

    /// Record a subscription's position. The webhook callback is kept unless `entry` names one.
    pub(super) fn set_sub(&self, key: &str, mut entry: PersistedSub) {
        {
            let mut d = lock(&self.inner.data);
            if entry.webhook.is_none() {
                entry.webhook = d.subs.get(key).and_then(|s| s.webhook.clone());
            }
            if d.subs.get(key) == Some(&entry) {
                return;
            }
            d.subs.insert(key.to_owned(), entry);
            d.snapshot_dirty = true;
        }
        self.dirty();
    }

    /// Record (or with `None`, drop) a webhook subscription's callback.
    pub(super) fn set_webhook(&self, key: &str, webhook: Option<PersistedWebhook>) {
        {
            let mut d = lock(&self.inner.data);
            let entry = d.subs.entry(key.to_owned()).or_default();
            if entry.webhook == webhook {
                return;
            }
            entry.webhook = webhook;
            d.snapshot_dirty = true;
        }
        self.dirty();
    }

    /// Every persisted webhook callback, as `(subscription key, token)`.
    pub(super) fn webhook_tokens(&self) -> Vec<(String, String)> {
        lock(&self.inner.data)
            .subs
            .iter()
            .filter_map(|(k, s)| s.webhook.as_ref().map(|w| (k.clone(), w.token.clone())))
            .collect()
    }

    /// An explicit unsubscribe forgets the position; a session ending keeps it, to resume from.
    pub(super) fn forget_sub(&self, key: &str) {
        {
            let mut d = lock(&self.inner.data);
            if d.subs.remove(key).is_none() {
                return;
            }
            d.snapshot_dirty = true;
        }
        self.dirty();
    }

    /// Queue an event for the model. `false` when the queue is at its count or byte bound: the
    /// caller must then *not* acknowledge or advance past it (backpressure). A single event larger
    /// than the byte bound is still taken into an empty queue, so it cannot wedge a subscription.
    pub(super) fn push_pending(&self, mut event: PendingEvent) -> bool {
        {
            let mut d = lock(&self.inner.data);
            event.seq = d.next_seq;
            event.batch = None;
            let op = LogOp::Add(event);
            let line = log_line(&op);
            let LogOp::Add(mut event) = op else {
                return false;
            };
            event.size = line.len() as u64;
            if d.pending.len() >= self.inner.max_pending
                || (!d.pending.is_empty() && d.pending_bytes + event.size > self.inner.max_bytes)
            {
                return false;
            }
            d.next_seq += 1;
            d.pending_bytes += event.size;
            d.pending.push(event);
            d.unwritten.extend_from_slice(&line);
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

    /// Put a batch's events back in the ready queue: its injection never reached the model.
    pub(super) fn unassign(&self, batch: u64) {
        let mut d = lock(&self.inner.data);
        for p in d.pending.iter_mut().filter(|p| p.batch == Some(batch)) {
            p.batch = None;
        }
        drop(d);
        self.inner.changed.notify_one();
    }

    /// A batch's injection did not reach the model (a failed run, a steer an abort dropped): its
    /// events are ready again — except those that have now failed `max_attempts` times, which are
    /// removed (a poison event must not re-run the model forever) and returned to be reported.
    pub(super) fn return_batch(&self, batch: u64, max_attempts: u32) -> Vec<PendingEvent> {
        let mut dropped = Vec::new();
        {
            let mut d = lock(&self.inner.data);
            let mut freed = 0;
            let mut seqs = Vec::new();
            d.pending.retain_mut(|p| {
                if p.batch != Some(batch) {
                    return true;
                }
                p.batch = None;
                p.attempts += 1;
                if p.attempts < max_attempts {
                    return true;
                }
                freed += p.size;
                seqs.push(p.seq);
                dropped.push(p.clone());
                false
            });
            if !seqs.is_empty() {
                d.pending_bytes = d.pending_bytes.saturating_sub(freed);
                let line = log_line(&LogOp::Done(seqs));
                d.unwritten.extend_from_slice(&line);
            }
        }
        if !dropped.is_empty() {
            self.dirty();
        }
        self.inner.changed.notify_one();
        dropped
    }

    /// The model has seen `batch` (its prompt reached the model and the transcript holding it is
    /// persisted).
    pub(super) fn delivered(&self, batch: u64) -> bool {
        let removed = {
            let mut d = lock(&self.inner.data);
            let mut seqs = Vec::new();
            let mut freed = 0;
            d.pending.retain(|p| {
                if p.batch == Some(batch) {
                    seqs.push(p.seq);
                    freed += p.size;
                    false
                } else {
                    true
                }
            });
            if !seqs.is_empty() {
                d.pending_bytes = d.pending_bytes.saturating_sub(freed);
                let line = log_line(&LogOp::Done(seqs));
                d.unwritten.extend_from_slice(&line);
                true
            } else {
                false
            }
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
            if added {
                d.snapshot_dirty = true;
            }
            added
        };
        if added {
            self.dirty();
        }
        added
    }

    /// Move the state to a new transcript's files, now: whatever is pending is flushed, whatever
    /// the target already holds (its own undelivered events, cursors, keys) is merged in — never
    /// overwritten — the merged state is written there, and the old files are removed. Returns once
    /// that is done (bounded).
    pub(super) async fn relocate(&self, path: Option<PathBuf>) {
        if lock(&self.inner.data).path == path {
            return;
        }
        let (tx, rx) = oneshot::channel();
        if self.inner.tx.send(Req::Relocate(path, tx)).is_ok() {
            let _ = tokio::time::timeout(COMMIT_TIMEOUT, rx).await;
        }
    }

    /// Wait for a write that includes every change made before this call. `Err` when it could not
    /// be made durable (a write error, or no answer within the timeout): the caller must not tell a
    /// server it was received. The changes stay queued and the next write retries them.
    pub(super) async fn commit(&self) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        self.inner
            .tx
            .send(Req::Commit(tx))
            .map_err(|_| "the events state writer is gone".to_owned())?;
        match tokio::time::timeout(COMMIT_TIMEOUT, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err("the events state writer is gone".to_owned()),
            Err(_) => Err(format!(
                "the events state was not written within {}s",
                COMMIT_TIMEOUT.as_secs()
            )),
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

/// What a state's files held.
#[derive(Default)]
struct Loaded {
    snapshot: Snapshot,
    pending: Vec<PendingEvent>,
    log_len: u64,
}

/// Read a state's snapshot and replay its log. Missing files are an empty state; a torn or
/// unparseable log line is skipped.
fn load(path: &Path) -> Loaded {
    let snapshot = std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice::<Snapshot>(&b).ok())
        .unwrap_or_default();
    let mut log = std::fs::read(log_path(path)).unwrap_or_default();
    // A torn tail — a crash mid-append — is cut off on disk before anything is appended again:
    // left in place, the next record would be written onto the end of the fragment, and the two
    // would be one unparseable line, losing an event that was acknowledged.
    let complete = log.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    if complete < log.len() {
        match truncate_durably(&log_path(path), complete as u64) {
            Ok(()) => log.truncate(complete),
            Err(e) => tracing::warn!(error = %e, "could not cut a torn MCP Events log tail"),
        }
    }
    let mut live: BTreeMap<u64, PendingEvent> = BTreeMap::new();
    for line in log.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        match serde_json::from_slice::<LogOp>(line) {
            Ok(LogOp::Add(mut e)) => {
                e.size = line.len() as u64 + 1;
                live.insert(e.seq, e);
            }
            Ok(LogOp::Done(seqs)) => {
                for s in seqs {
                    live.remove(&s);
                }
            }
            Err(_) => {}
        }
    }
    Loaded {
        snapshot,
        pending: live.into_values().collect(),
        log_len: log.len() as u64,
    }
}

async fn writer(
    data: Arc<Mutex<Data>>,
    mut rx: mpsc::UnboundedReceiver<Req>,
    loaded: watch::Sender<bool>,
    path: Option<PathBuf>,
    changed: Arc<Notify>,
) {
    if let Some(path) = path
        && let Ok(l) = tokio::task::spawn_blocking(move || load(&path)).await
    {
        let mut d = lock(&data);
        d.next_seq = l.pending.iter().map(|p| p.seq).max().unwrap_or(0) + 1;
        d.pending_bytes = l.pending.iter().map(|p| p.size).sum();
        d.subs = l.snapshot.subscriptions;
        d.server_keys = l.snapshot.server_keys;
        d.pending = l.pending;
        d.log_len = l.log_len;
    }
    let _ = loaded.send(true);
    changed.notify_one();
    loop {
        let Some(first) = rx.recv().await else {
            // Every handle is gone: one last write, then end.
            let _ = flush(&data).await;
            return;
        };
        let mut commits = Vec::new();
        let mut relocations = Vec::new();
        let mut closed = false;
        match first {
            Req::Commit(tx) => commits.push(tx),
            Req::Relocate(p, tx) => relocations.push((p, tx)),
            Req::Dirty => {
                // Coalesce ordinary changes for up to 100 ms — but someone waiting on a commit (a
                // webhook ack) is never made to sit out the window.
                let window = tokio::time::sleep(Duration::from_millis(100));
                tokio::pin!(window);
                loop {
                    tokio::select! {
                        () = &mut window => break,
                        req = rx.recv() => match req {
                            Some(Req::Dirty) => {}
                            Some(Req::Commit(tx)) => {
                                commits.push(tx);
                                break;
                            }
                            Some(Req::Relocate(p, tx)) => {
                                relocations.push((p, tx));
                                break;
                            }
                            None => {
                                closed = true;
                                break;
                            }
                        },
                    }
                }
            }
        }
        while !closed {
            match rx.try_recv() {
                Ok(Req::Commit(tx)) => commits.push(tx),
                Ok(Req::Relocate(p, tx)) => relocations.push((p, tx)),
                Ok(Req::Dirty) => {}
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    closed = true;
                    break;
                }
            }
        }
        let result = flush(&data).await;
        for tx in commits {
            let _ = tx.send(result.clone());
        }
        for (target, tx) in relocations {
            relocate(&data, target).await;
            let _ = tx.send(());
        }
        if closed {
            return;
        }
    }
}

/// Write what changed: the log's new records (or, when it has grown mostly dead, the whole live
/// log, compacted), then the snapshot if it changed. On failure everything taken is put back for
/// the next write.
async fn flush(data: &Arc<Mutex<Data>>) -> Result<(), String> {
    enum LogWrite {
        None,
        Append(Vec<u8>),
        /// The whole live log, and the unwritten records it makes redundant — kept to be put back
        /// if the rewrite fails, so they still reach the disk by the next write, compacted or not.
        Rewrite(Vec<u8>, Vec<u8>),
    }
    let (path, log_write, snapshot) = {
        let mut d = lock(data);
        let Some(path) = d.path.clone() else {
            d.unwritten.clear();
            d.snapshot_dirty = false;
            return Ok(());
        };
        let compact = d.log_len + d.unwritten.len() as u64 > COMPACT_MIN_BYTES
            && d.log_len + d.unwritten.len() as u64 > 2 * d.pending_bytes;
        let log_write = if compact {
            let displaced = std::mem::take(&mut d.unwritten);
            let mut all = Vec::with_capacity(d.pending_bytes as usize);
            for e in &d.pending {
                all.extend_from_slice(&log_line(&LogOp::Add(e.clone())));
            }
            LogWrite::Rewrite(all, displaced)
        } else if d.unwritten.is_empty() {
            LogWrite::None
        } else {
            LogWrite::Append(std::mem::take(&mut d.unwritten))
        };
        let snapshot = d.snapshot_dirty.then(|| {
            d.snapshot_dirty = false;
            Snapshot {
                spec_commit: SPEC_COMMIT.to_owned(),
                subscriptions: d.subs.clone(),
                server_keys: d.server_keys.clone(),
            }
        });
        (path, log_write, snapshot)
    };
    if matches!(log_write, LogWrite::None) && snapshot.is_none() {
        return Ok(());
    }
    let task_path = path.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        let log = log_path(&task_path);
        let written = match &log_write {
            LogWrite::None => Ok(None),
            LogWrite::Append(bytes) => {
                append_durably(&log, bytes).map(|()| Some((false, bytes.len() as u64)))
            }
            LogWrite::Rewrite(bytes, _) => {
                write_atomic(&log, bytes).map(|()| Some((true, bytes.len() as u64)))
            }
        };
        let log_done = match written {
            Ok(done) => done,
            // The snapshot is not written either: a cursor must not run ahead of its events.
            Err(e) => return (Some(log_write), snapshot, Err(e.to_string()), None),
        };
        let result = match &snapshot {
            Some(s) => serde_json::to_vec(s)
                .map_err(|e| e.to_string())
                .and_then(|b| write_atomic(&task_path, &b).map_err(|e| e.to_string())),
            None => Ok(()),
        };
        (None, snapshot, result, log_done)
    })
    .await;
    let (put_back, snapshot, result, log_done) = match outcome {
        Ok(v) => v,
        Err(e) => return Err(format!("the events state writer failed: {e}")),
    };
    let mut d = lock(data);
    if d.path.as_deref() != Some(path.as_path()) {
        return result;
    }
    // A failed append, or a failed compaction's displaced records, go back in front of whatever was
    // added meanwhile: the next write retries them, whether it compacts or appends.
    if let Some(LogWrite::Append(mut bytes) | LogWrite::Rewrite(_, mut bytes)) = put_back {
        bytes.extend_from_slice(&d.unwritten);
        d.unwritten = bytes;
    }
    if let Some((rewrite, n)) = log_done {
        d.log_len = if rewrite { n } else { d.log_len + n };
    }
    if result.is_err() && snapshot.is_some() {
        d.snapshot_dirty = true;
    }
    if let Err(e) = &result {
        tracing::warn!(error = %e, "could not persist MCP Events state; will retry");
    }
    result
}

/// The relocation itself (see [`StateStore::relocate`]), run by the writer after a flush.
async fn relocate(data: &Arc<Mutex<Data>>, target: Option<PathBuf>) {
    let old = lock(data).path.clone();
    if old == target {
        return;
    }
    let Some(target) = target else {
        lock(data).path = None;
        return;
    };
    let read_target = target.clone();
    let theirs = tokio::task::spawn_blocking(move || load(&read_target))
        .await
        .unwrap_or_default();
    let (snapshot, log) = {
        let mut d = lock(data);
        for (k, v) in theirs.snapshot.subscriptions {
            d.subs.entry(k).or_insert(v);
        }
        for (origin, keys) in theirs.snapshot.server_keys {
            let set = d.server_keys.entry(origin).or_default();
            for k in keys {
                if !set.contains(&k) {
                    set.push(k);
                }
            }
        }
        for mut e in theirs.pending {
            e.seq = d.next_seq;
            d.next_seq += 1;
            e.batch = None;
            e.size = log_line(&LogOp::Add(e.clone())).len() as u64;
            d.pending_bytes += e.size;
            d.pending.push(e);
        }
        d.path = Some(target.clone());
        d.unwritten.clear();
        d.snapshot_dirty = false;
        if d.is_empty() {
            (None, None)
        } else {
            let mut log = Vec::with_capacity(d.pending_bytes as usize);
            for e in &d.pending {
                log.extend_from_slice(&log_line(&LogOp::Add(e.clone())));
            }
            d.log_len = log.len() as u64;
            (
                Some(Snapshot {
                    spec_commit: SPEC_COMMIT.to_owned(),
                    subscriptions: d.subs.clone(),
                    server_keys: d.server_keys.clone(),
                }),
                Some(log),
            )
        }
    };
    let written = tokio::task::spawn_blocking(move || -> std::io::Result<()> {
        if let (Some(s), Some(log)) = (snapshot, log) {
            write_atomic(&log_path(&target), &log)?;
            write_atomic(&target, &serde_json::to_vec(&s)?)?;
        }
        if let Some(old) = old {
            let _ = std::fs::remove_file(log_path(&old));
            let _ = std::fs::remove_file(&old);
        }
        Ok(())
    })
    .await;
    if !matches!(written, Ok(Ok(()))) {
        tracing::warn!("could not move the MCP Events state to the new transcript; will retry");
        let mut d = lock(data);
        d.snapshot_dirty = true;
        // Rewrite the whole log at the next flush.
        d.log_len = u64::MAX / 4;
    }
}

fn private_open(path: &Path, append: bool) -> std::io::Result<std::fs::File> {
    let mut opts = std::fs::OpenOptions::new();
    if append {
        opts.append(true).create(true);
    } else {
        opts.write(true).create_new(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        opts.mode(0o600);
    }
    opts.open(path)
}

fn sync_dir(path: &Path) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::File::open(dir)?.sync_all()?;
    }
    Ok(())
}

/// Cut a file back to `len` bytes, durably.
fn truncate_durably(path: &Path, len: u64) -> std::io::Result<()> {
    let f = std::fs::OpenOptions::new().write(true).open(path)?;
    f.set_len(len)?;
    f.sync_all()
}

/// Append `bytes` and `fdatasync` them (and the directory, the first time the file appears).
fn append_durably(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;
    let existed = path.exists();
    let mut f = private_open(path, true)?;
    f.write_all(bytes)?;
    f.sync_data()?;
    if !existed {
        sync_dir(path)?;
    }
    Ok(())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;
    let tmp = path.with_extension(format!(
        "{}.{}.tmp",
        path.extension().and_then(|e| e.to_str()).unwrap_or("x"),
        crate::tools::temp_suffix()
    ));
    let write = (|| -> std::io::Result<()> {
        let mut f = private_open(&tmp, false)?;
        f.write_all(bytes)?;
        f.flush()?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&tmp, path)?;
        sync_dir(path)
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
        event_sized(n, 0)
    }

    fn event_sized(n: u64, pad: usize) -> PendingEvent {
        PendingEvent::new(
            McpEventAction::FollowUp,
            "s".into(),
            "n".into(),
            serde_json::json!({}),
            None,
            serde_json::json!({ "eventId": format!("e{n}"), "data": "x".repeat(pad) }),
        )
    }

    fn on_disk(path: &Path) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }

    fn store(path: &Path, max: usize) -> StateStore {
        StateStore::open(Some(path.to_path_buf()), max, u64::MAX)
    }

    #[tokio::test]
    async fn dropping_every_handle_writes_once_more_and_ends_the_writer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.mcp-events.json");
        let store = store(&path, 10);
        store.loaded().await;
        store.set_sub(
            "k",
            PersistedSub {
                cursor: Some("7".into()),
                recent: vec!["a".into()],
                webhook: None,
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
        let store = store(&path, 10);
        store.loaded().await;
        for i in 0..200 {
            store.set_sub(
                "k",
                PersistedSub {
                    cursor: Some(i.to_string()),
                    ..PersistedSub::default()
                },
            );
            if i % 37 == 0 {
                store.commit().await.unwrap();
            }
        }
        store.commit().await.unwrap();
        assert_eq!(on_disk(&path)["subscriptions"]["k"]["cursor"], "199");
    }

    #[tokio::test]
    async fn pending_events_survive_a_reopen_ready_again_and_backpressure_at_the_cap() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.mcp-events.json");
        let s = store(&path, 2);
        s.loaded().await;
        assert!(s.push_pending(event(1)));
        assert!(s.push_pending(event(2)));
        assert!(
            !s.push_pending(event(3)),
            "at the cap the event is refused, not dropped"
        );
        let ready = s.ready_pending();
        s.assign_batch(&[ready[0].seq], 9);
        assert_eq!(s.ready_pending().len(), 1);
        s.commit().await.unwrap();
        drop(s);

        let again = store(&path, 2);
        again.loaded().await;
        let ready = again.ready_pending();
        assert_eq!(
            ready.len(),
            2,
            "an in-flight batch is ready again after a restart"
        );
        assert_eq!(ready[0].event["eventId"], "e1");
        again.assign_batch(&[ready[0].seq], 1);
        assert!(again.delivered(1));
        again.commit().await.unwrap();
        drop(again);
        let third = store(&path, 2);
        third.loaded().await;
        let ready = third.ready_pending();
        assert_eq!(ready.len(), 1, "a delivered event stays delivered");
        assert_eq!(ready[0].event["eventId"], "e2");
    }

    /// The queue is bounded in bytes, not only in count.
    #[tokio::test]
    async fn the_pending_queue_is_bounded_in_bytes() {
        let s = StateStore::open(None, 1000, 1024 * 1024);
        s.loaded().await;
        let mut taken = 0;
        while s.push_pending(event_sized(taken, 256 * 1024)) {
            taken += 1;
            assert!(taken < 100, "never refused");
        }
        assert_eq!(taken, 3, "four 256 KiB events exceed a 1 MiB bound");
        // A single event larger than the bound still fits an empty queue.
        let big = StateStore::open(None, 1000, 1024);
        big.loaded().await;
        assert!(big.push_pending(event_sized(0, 4096)));
        assert!(!big.push_pending(event(1)));
    }

    /// A commit writes what changed, not the queue: with 200 large events pending, the snapshot stays
    /// tiny and each commit appends one event's worth to the log.
    #[tokio::test]
    async fn commit_cost_tracks_the_change_not_the_queue() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.mcp-events.json");
        let s = store(&path, 10_000);
        s.loaded().await;
        s.set_sub("k", PersistedSub::default());
        let mut last = 0u64;
        for i in 0..200 {
            assert!(s.push_pending(event_sized(i, 64 * 1024)));
            s.set_sub(
                "k",
                PersistedSub {
                    cursor: Some(i.to_string()),
                    ..PersistedSub::default()
                },
            );
            s.commit().await.unwrap();
            let log = std::fs::metadata(log_path(&path)).unwrap().len();
            let grew = log - last;
            assert!(
                (64 * 1024..80 * 1024).contains(&grew),
                "commit {i} wrote {grew} bytes of log for one 64 KiB event"
            );
            last = log;
            assert!(
                std::fs::metadata(&path).unwrap().len() < 4096,
                "the snapshot holds no events"
            );
        }
        drop(s);
        let again = store(&path, 10_000);
        again.loaded().await;
        assert_eq!(again.pending_len(), 200);
    }

    /// Delivered events are compacted out of the log once they dominate it.
    #[tokio::test]
    async fn the_log_is_compacted_once_mostly_dead() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.mcp-events.json");
        let s = store(&path, 10_000);
        s.loaded().await;
        for i in 0..40 {
            assert!(s.push_pending(event_sized(i, 64 * 1024)));
        }
        let seqs: Vec<u64> = s.ready_pending().iter().map(|e| e.seq).collect();
        s.assign_batch(&seqs[..39], 1);
        s.commit().await.unwrap();
        assert!(s.delivered(1));
        s.commit().await.unwrap();
        let log = std::fs::metadata(log_path(&path)).unwrap().len();
        assert!(
            log < 2 * 70 * 1024,
            "compacted to the one live event: {log}"
        );
        drop(s);
        let again = store(&path, 10_000);
        again.loaded().await;
        assert_eq!(again.pending_len(), 1);
    }

    /// A write that fails is reported, and retried by the next one — nothing is lost.
    #[tokio::test]
    async fn a_failed_write_is_reported_and_retried() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = dir.path().join("sessions");
        std::fs::create_dir(&sessions).unwrap();
        let path = sessions.join("s.mcp-events.json");
        let s = store(&path, 10);
        s.loaded().await;
        // The log's place is taken by a directory: appends fail.
        std::fs::create_dir(log_path(&path)).unwrap();
        assert!(s.push_pending(event(1)));
        assert!(s.commit().await.is_err(), "the failure is reported");
        std::fs::remove_dir(log_path(&path)).unwrap();
        assert!(s.push_pending(event(2)));
        s.commit().await.unwrap();
        drop(s);
        let again = store(&path, 10);
        again.loaded().await;
        assert_eq!(again.pending_len(), 2, "the failed append was retried");
    }

    #[tokio::test]
    async fn nothing_is_written_while_there_is_nothing_to_persist() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.mcp-events.json");
        let s = store(&path, 10);
        s.loaded().await;
        s.commit().await.unwrap();
        s.forget_sub("nothing");
        drop(s);
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
    }

    #[tokio::test]
    async fn relocating_merges_into_the_target_and_removes_the_old_files() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.mcp-events.json");
        let b = dir.path().join("b.mcp-events.json");
        // `b` already has state of its own: an undelivered event and a cursor.
        {
            let other = store(&b, 10);
            other.loaded().await;
            assert!(other.push_pending(event(100)));
            other.set_sub(
                "theirs",
                PersistedSub {
                    cursor: Some("t".into()),
                    ..PersistedSub::default()
                },
            );
            other.commit().await.unwrap();
        }
        let s = store(&a, 10);
        s.loaded().await;
        assert!(s.add_keys("https://x", &["k1".into()]));
        assert!(!s.add_keys("https://x", &["k1".into()]));
        assert!(s.push_pending(event(1)));
        s.set_sub("ours", PersistedSub::default());
        s.relocate(Some(b.clone())).await;
        assert!(
            !a.exists() && !log_path(&a).exists(),
            "the old files are gone"
        );
        assert_eq!(on_disk(&b)["server_keys"]["https://x"][0], "k1");
        assert_eq!(on_disk(&b)["subscriptions"]["theirs"]["cursor"], "t");
        assert!(on_disk(&b)["subscriptions"]["ours"].is_object());
        drop(s);
        let again = store(&b, 10);
        again.loaded().await;
        let ids: Vec<String> = again
            .ready_pending()
            .iter()
            .map(|e| e.event["eventId"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(
            ids,
            ["e1", "e100"],
            "neither side's undelivered events lost"
        );
    }

    /// A crash mid-append leaves half a record at the end of the log. The next acknowledged event
    /// must not be glued onto it (and lost on every later replay): the torn tail is cut off first.
    #[tokio::test]
    async fn a_torn_log_tail_does_not_swallow_the_next_event() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.mcp-events.json");
        {
            let s = store(&path, 10);
            s.loaded().await;
            assert!(s.push_pending(event(1)));
            s.commit().await.unwrap();
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        // The crash: half of a second record.
        {
            use std::io::Write as _;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(log_path(&path))
                .unwrap();
            f.write_all(br#"{"add":{"seq":2,"action":"follow_up","ser"#)
                .unwrap();
        }
        {
            let s = store(&path, 10);
            s.loaded().await;
            assert_eq!(s.pending_len(), 1, "the torn record is not an event");
            let mut after = event(0);
            after.event = serde_json::json!({ "eventId": "after-crash" });
            assert!(s.push_pending(after));
            s.commit().await.unwrap();
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        let again = store(&path, 10);
        again.loaded().await;
        let ids: Vec<String> = again
            .ready_pending()
            .iter()
            .map(|e| e.event["eventId"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(
            ids,
            ["e1", "after-crash"],
            "the event after the crash survives"
        );
    }

    /// A compaction that fails puts back the records it would have made redundant: if the next
    /// write appends instead of compacting, they still reach the disk.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_failed_compaction_loses_no_records() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = tempfile::tempdir().unwrap();
        let sessions = dir.path().join("sessions");
        std::fs::create_dir(&sessions).unwrap();
        let path = sessions.join("s.mcp-events.json");
        let s = store(&path, 10_000);
        s.loaded().await;
        // ~2 MB of log, all but one event delivered: the next write wants to compact.
        for i in 0..30 {
            assert!(s.push_pending(event_sized(i, 64 * 1024)));
        }
        s.commit().await.unwrap();
        let seqs: Vec<u64> = s.ready_pending().iter().map(|e| e.seq).collect();
        s.assign_batch(&seqs[..29], 1);
        assert!(s.delivered(1));
        let mut x = event(0);
        x.event = serde_json::json!({ "eventId": "x" });
        assert!(s.push_pending(x));
        // The compaction's temp file cannot be created; appends to the existing log still work.
        std::fs::set_permissions(&sessions, std::fs::Permissions::from_mode(0o500)).unwrap();
        if std::fs::write(sessions.join("probe"), b"").is_ok() {
            // Running as root: permissions do not bind, the failure cannot be staged.
            std::fs::set_permissions(&sessions, std::fs::Permissions::from_mode(0o700)).unwrap();
            return;
        }
        let failed = s.commit().await;
        std::fs::set_permissions(&sessions, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(failed.is_err(), "the compaction failed");
        // Now enough live data that the next write appends rather than compacts.
        for i in 100..140 {
            assert!(s.push_pending(event_sized(i, 64 * 1024)));
        }
        s.commit().await.unwrap();
        drop(s);
        tokio::time::sleep(Duration::from_millis(100)).await;
        let again = store(&path, 10_000);
        again.loaded().await;
        let ids: Vec<String> = again
            .ready_pending()
            .iter()
            .map(|e| e.event["eventId"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(
            ids.len(),
            42,
            "1 + x + 40, and the 29 delivered stay delivered: {}",
            ids.len()
        );
        assert!(
            ids.contains(&"x".to_owned()),
            "the event pushed before the failure is on disk"
        );
    }
}
