//! JSONL session persistence: a per-session append log and a multi-session repository.
//!
//! Each session is one newline-delimited JSON file: a header line carrying [`SessionMeta`] (stable id,
//! cwd, model, timestamps, optional title and `parent` lineage), then one line per conversation
//! message. A turn **appends** its new messages — O(new), not O(transcript) — and a torn final line
//! (a crash mid-append) is dropped on load, so at most the last entry is lost. Compaction rewrites the
//! whole file atomically (temp + rename), since it rewrites the message list rather than extending it.
//!
//! A [`SessionRepo`] is a directory of those files: it lists sessions (newest first), creates and
//! opens them, deletes, and **forks** one (copy its prefix into a new session that links back via
//! `parent`) — the headless-relevant slice of pi's session tree.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use agent_core::{ContentBlock, Message, Role, Session};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// On-disk format version, written into every header. The load path refuses any header whose version
/// is *newer* than this — a forward-compat guard so an older binary never silently mis-parses a shape
/// it doesn't understand — and treats `0` (a pre-versioning header, where the field was absent and
/// `serde` defaulted it) as v1-equivalent. The contract: additive, optional fields (each `#[serde(default)]`)
/// do **not** need a bump, since old and new files round-trip through each other; only an incompatible
/// reshaping of existing fields does. When that happens, bump this and add a [`migrate`] arm that
/// upgrades the older shape in place.
const VERSION: u32 = 1;

/// How many characters of the first user message a listing preview keeps.
const PREVIEW_MAX: usize = 80;

/// Stable identity + metadata for one session, persisted as the file's header line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    /// On-disk format version (see [`VERSION`]) — present so a future migration can branch on it.
    #[serde(default)]
    pub version: u32,
    pub id: String,
    /// Unix seconds at creation. Orders the repo listing.
    pub created_at: u64,
    /// Working directory the session was started in.
    pub cwd: String,
    /// Model the session runs against.
    pub model: String,
    /// Human-readable title, if set.
    #[serde(default)]
    pub title: Option<String>,
    /// The session this was forked from, if any.
    #[serde(default)]
    pub parent: Option<String>,
    /// Compaction provenance: how many times this session was rewritten with *fewer* messages than
    /// were on disk (i.e. a compaction replaced a prefix with a summary). Defaults to 0 for sessions
    /// that were never compacted, so older headers round-trip unchanged.
    #[serde(default)]
    pub compactions: u32,
    /// Total messages dropped across those compactions — a coarse measure of how much transcript the
    /// summaries stand in for. Defaults to 0.
    #[serde(default)]
    pub dropped_messages: u64,

    // --- Derived listing fields ---
    // Populated only by [`SessionRepo::list`] (from the file's mtime and a light scan), never persisted
    // — `#[serde(skip)]` keeps them out of the on-disk header so they can't go stale, and defaults them
    // to zero on a freshly created or opened session. They let a client sort by last-active and show a
    // preview without opening every transcript.
    /// Unix seconds of the file's last modification — when the session was last written to. `0` outside
    /// of a listing. Sort by this for "most recently active" ordering; `created_at` only orders by birth.
    #[serde(skip)]
    pub updated_at: u64,
    /// Number of message lines on disk (a torn final line is excluded, matching load semantics). `0`
    /// outside of a listing.
    #[serde(skip)]
    pub message_count: usize,
    /// The first user message's text, truncated to [`PREVIEW_MAX`] chars — `None` outside of a listing,
    /// or when the session has no user text yet (e.g. empty, or only tool-result turns so far).
    #[serde(skip)]
    pub preview: Option<String>,
}

impl SessionMeta {
    /// Fresh metadata with a generated id and the current timestamp.
    pub fn new(cwd: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            version: VERSION,
            id: new_id(),
            created_at: now_secs(),
            cwd: cwd.into(),
            model: model.into(),
            title: None,
            parent: None,
            compactions: 0,
            dropped_messages: 0,
            updated_at: 0,
            message_count: 0,
            preview: None,
        }
    }
}

/// One persisted line. Internally tagged on `type`, so a `Message` (which serializes as
/// `{role, content}`) is stored as `{"type":"message", role, content}`.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Entry {
    Session(SessionMeta),
    Message(Message),
}

/// A handle to one session's append log.
pub struct SessionStore {
    path: PathBuf,
    meta: SessionMeta,
    /// How many messages are already on disk — the append cursor.
    persisted: usize,
}

impl SessionStore {
    /// Create a new session file at `path`, writing its header. Errors if the file already exists.
    pub fn create(path: PathBuf, meta: SessionMeta) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        write_line(&mut f, &Entry::Session(meta.clone()))?;
        // Durability: get the header bytes (and the new file's directory entry) onto stable storage
        // before returning, so a crash right after `create` can't lose the session entirely.
        f.flush()?;
        f.sync_all()?;
        fsync_dir(&path)?;
        Ok(Self {
            path,
            meta,
            persisted: 0,
        })
    }

    /// Open an existing session file, returning the store and the restored [`Session`]. A torn final
    /// line (crash mid-append) is skipped; a header is required.
    pub fn open(path: PathBuf) -> std::io::Result<(Self, Session)> {
        let file = File::open(&path)?;
        let mut meta: Option<SessionMeta> = None;
        let mut messages: Vec<Message> = Vec::new();
        for line in BufReader::new(file).lines() {
            // A torn write makes `lines()` yield an `Err` (or a partial line that won't parse); either
            // way we stop at the first unreadable line — only the last entry can be lost.
            let Ok(line) = line else { break };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<Entry>(line) {
                Ok(Entry::Session(m)) => meta = Some(m),
                Ok(Entry::Message(msg)) => messages.push(msg),
                // Unparseable (torn) line — stop; nothing valid follows a half-written record.
                Err(_) => break,
            }
        }
        let meta = meta.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "session file has no header",
            )
        })?;
        // Validate (and, in a future format, upgrade) the header before trusting the rest of the file.
        let meta = migrate(meta, &path)?;
        let persisted = messages.len();
        let mut session = Session::new();
        session.messages = Arc::new(messages);
        Ok((
            Self {
                path,
                meta,
                persisted,
            },
            session,
        ))
    }

    /// The session's metadata.
    pub fn meta(&self) -> &SessionMeta {
        &self.meta
    }

    /// Append the messages added since the last persist. O(new messages).
    pub fn append_new(&mut self, messages: &[Message]) -> std::io::Result<()> {
        if messages.len() <= self.persisted {
            return Ok(());
        }
        let mut f = OpenOptions::new().append(true).open(&self.path)?;
        for msg in &messages[self.persisted..] {
            write_line(&mut f, &Entry::Message(msg.clone()))?;
        }
        // `flush` only pushes past our buffer into the OS; `sync_all` forces the bytes to disk, which
        // is what the module's crash-safety claim actually requires. The parent dir is unchanged on an
        // append (same inode, same dentry), so no directory fsync is needed here.
        f.flush()?;
        f.sync_all()?;
        self.persisted = messages.len();
        Ok(())
    }

    /// Rewrite the whole file atomically (header + every message). Used when the message list was
    /// rewritten rather than extended — e.g. after compaction replaced the prefix with a summary.
    pub fn rewrite(&mut self, messages: &[Message]) -> std::io::Result<()> {
        // Compaction provenance: a rewrite that ends with *fewer* messages than were on disk dropped a
        // prefix (the compaction case), so record it in the header before writing. A same-length rewrite
        // (e.g. `set_title`) or a fresh fork (`persisted == 0`) drops nothing and records nothing.
        let dropped = self.persisted.saturating_sub(messages.len());
        if dropped > 0 {
            self.meta.compactions = self.meta.compactions.saturating_add(1);
            self.meta.dropped_messages = self.meta.dropped_messages.saturating_add(dropped as u64);
        }

        let tmp = self.path.with_extension("jsonl.tmp");
        let mut f = File::create(&tmp)?;
        write_line(&mut f, &Entry::Session(self.meta.clone()))?;
        for msg in messages {
            write_line(&mut f, &Entry::Message(msg.clone()))?;
        }
        // Sync the temp file's contents, then rename (atomic), then fsync the parent directory so the
        // rename itself is durable: without the dir fsync a crash could surface the old file — or, in the
        // window between, neither — even though the new bytes had reached disk.
        f.flush()?;
        f.sync_all()?;
        fs::rename(&tmp, &self.path)?;
        fsync_dir(&self.path)?;
        self.persisted = messages.len();
        Ok(())
    }

    /// Set (and persist) the session title. The title lives in the header so the repo listing can read
    /// it cheaply, so this rewrites the file (titles change rarely); pass the current messages.
    pub fn set_title(
        &mut self,
        title: impl Into<String>,
        messages: &[Message],
    ) -> std::io::Result<()> {
        self.meta.title = Some(title.into());
        self.rewrite(messages)
    }
}

/// A directory of session files.
pub struct SessionRepo {
    dir: PathBuf,
}

impl SessionRepo {
    /// Open (creating if needed) a repository rooted at `dir`.
    pub fn open(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    fn path_for(&self, meta: &SessionMeta) -> PathBuf {
        self.dir
            .join(format!("{}_{}.jsonl", meta.created_at, meta.id))
    }

    /// Create a new, empty session and return its store.
    pub fn create(&self, meta: SessionMeta) -> std::io::Result<SessionStore> {
        SessionStore::create(self.path_for(&meta), meta)
    }

    /// All sessions' metadata, newest first (by `created_at`; clients can re-sort by `updated_at`). Each
    /// entry carries the derived listing fields (`updated_at`, `message_count`, `preview`). Files that
    /// fail to read, lack a header, or carry an unreadable version are skipped.
    pub fn list(&self) -> std::io::Result<Vec<SessionMeta>> {
        let mut metas = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            if let Some(meta) = read_listing(&path) {
                metas.push(meta);
            }
        }
        metas.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(metas)
    }

    /// Open a session by id.
    pub fn open_id(&self, id: &str) -> std::io::Result<(SessionStore, Session)> {
        let path = self.find_path(id).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, format!("no session {id}"))
        })?;
        SessionStore::open(path)
    }

    /// Delete a session by id. Idempotent per the repo invariant "check before destroy; don't error if
    /// it's gone": deleting an absent (or already-deleted) session is a successful no-op.
    pub fn delete(&self, id: &str) -> std::io::Result<()> {
        let Some(path) = self.find_path(id) else {
            return Ok(());
        };
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            // Raced with another deleter: the file vanished between `find_path` and `remove`. Still a
            // no-op success — the post-condition ("no session with this id") holds either way.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Fork session `id` at `upto` messages: a new session whose transcript is the first `upto`
    /// messages of the original, linked back via `parent`. `upto` is clamped to the source length, so
    /// `usize::MAX` clones the whole session. Returns the new store and its restored session.
    pub fn fork(&self, id: &str, upto: usize) -> std::io::Result<(SessionStore, Session)> {
        let (src, src_session) = self.open_id(id)?;
        let upto = upto.min(src_session.messages.len());
        let mut meta = SessionMeta::new(src.meta.cwd.clone(), src.meta.model.clone());
        meta.parent = Some(id.to_string());
        meta.title = src.meta.title.clone();

        let mut store = self.create(meta)?;
        let prefix: Vec<Message> = src_session.messages[..upto].to_vec();
        store.rewrite(&prefix)?;
        let mut session = Session::new();
        session.messages = Arc::new(prefix);
        Ok((store, session))
    }

    fn find_path(&self, id: &str) -> Option<PathBuf> {
        fs::read_dir(&self.dir).ok()?.flatten().find_map(|e| {
            let path = e.path();
            let name = path.file_name()?.to_str()?;
            // `<created_at>_<id>.jsonl`
            (name.ends_with(&format!("_{id}.jsonl"))).then_some(path)
        })
    }
}

/// Validate — and, in a future format, migrate — a header to the current [`VERSION`]. Today there is
/// exactly one readable shape; a header whose version is *newer* than this build understands means we'd
/// be guessing at fields we don't know, so we refuse it loudly rather than mis-parse. The `match` is the
/// extension point: a later bump keeps the newer-version guard and adds an arm per older version that
/// upgrades it in place.
fn migrate(meta: SessionMeta, path: &Path) -> std::io::Result<SessionMeta> {
    match meta.version {
        // `0` is a pre-versioning header (the field was absent and defaulted); it is wire-compatible
        // with v1. `VERSION` is the current shape.
        0 | VERSION => Ok(meta),
        v => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "session {} is format version {v}, which this build (version {VERSION}) cannot read",
                path.display()
            ),
        )),
    }
}

/// Read a session file's listing metadata: its (version-checked) header with the derived `updated_at` /
/// `message_count` / `preview` fields filled in. One streaming pass — lines are read and parsed
/// individually, never collected — so this stays light even for long transcripts (the header alone gives
/// id/title/etc.; only the count and preview need the scan). Returns `None` for a file that isn't a
/// readable session (no/invalid header, or an unreadable version), matching `list`'s skip semantics.
fn read_listing(path: &Path) -> Option<SessionMeta> {
    let updated_at = mtime_secs(path);
    let file = File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let mut buf = String::new();

    // The header is the first line.
    reader.read_line(&mut buf).ok()?;
    let mut meta = match serde_json::from_str::<Entry>(buf.trim()).ok()? {
        Entry::Session(m) => migrate(m, path).ok()?,
        Entry::Message(_) => return None,
    };

    let mut message_count = 0usize;
    let mut preview = None;
    loop {
        buf.clear();
        match reader.read_line(&mut buf) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(_) => break,
        }
        let line = buf.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<Entry>(line) {
            Ok(Entry::Message(msg)) => {
                message_count += 1;
                if preview.is_none() {
                    if let Some(text) = first_user_text(&msg) {
                        preview = Some(preview_of(text));
                    }
                }
            }
            // A stray header mid-file is ignored; a torn (unparseable) final line ends the scan, same as
            // the load path — only the last entry can be lost.
            Ok(Entry::Session(_)) => {}
            Err(_) => break,
        }
    }

    meta.updated_at = updated_at;
    meta.message_count = message_count;
    meta.preview = preview;
    Some(meta)
}

/// The first plain-text block of a user message — what a preview shows. Tool-result user turns carry no
/// `Text` block, so they yield `None` and are skipped; assistant turns aren't user input.
fn first_user_text(msg: &Message) -> Option<&str> {
    if msg.role != Role::User {
        return None;
    }
    msg.content.iter().find_map(|b| match b {
        ContentBlock::Text { text } => Some(text.as_str()),
        _ => None,
    })
}

/// A one-line, length-bounded preview: the first non-blank line of `text`, trimmed, truncated on a char
/// boundary to [`PREVIEW_MAX`] with an ellipsis when it overran.
fn preview_of(text: &str) -> String {
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    let mut out: String = line.chars().take(PREVIEW_MAX).collect();
    if line.chars().count() > PREVIEW_MAX {
        out.push('…');
    }
    out
}

/// The file's last-modified time as Unix seconds, read from `fs` metadata (no file content). Falls back
/// to 0 if the platform/clock can't supply it — a listing field, not a correctness input.
fn mtime_secs(path: &Path) -> u64 {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// fsync a path's parent directory so a just-created or just-renamed entry in it is itself durable
/// (a file `sync_all` persists contents, not the dentry that names them). A best-effort no-op if the
/// path has no parent.
fn fsync_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        // Opening the directory read-only and `sync_all`ing it flushes its updated entries. Linux/macOS
        // allow fsync on a directory handle; this is the standard atomic-rename durability step.
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

/// Serialize one entry as a single JSON line (no embedded newlines — `serde_json` escapes them).
fn write_line(w: &mut impl Write, entry: &Entry) -> std::io::Result<()> {
    let v = serde_json::to_value(entry).map_err(std::io::Error::other)?;
    // Defensive: a serialized line must be one physical line.
    debug_assert!(!serde_json::to_string(&v).unwrap_or_default().contains('\n'));
    writeln!(w, "{}", Value::to_string(&v))
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A collision-resistant, roughly time-ordered session id: `<nanos>-<salt>-<seq>`.
///
/// The old `{nanos}{seq}` scheme was only process-local — two processes that started in the same
/// nanosecond produced the same id. We splice in a per-process **salt** seeded once from `RandomState`
/// (the OS-seeded hasher `HashMap` uses), so independent processes diverge even at the same instant;
/// within a process the monotonic `seq` breaks same-nanosecond ties. The nanosecond prefix keeps ids
/// sortable by creation. A UUID-shaped string isn't required — global uniqueness and ordering are.
fn new_id() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::BuildHasher;
    use std::sync::atomic::{AtomicU64, Ordering};

    // Seeded once per process. `RandomState` draws OS entropy at construction, so hashing a fixed value
    // with it yields a value that's stable for this process but random across processes.
    static SALT: OnceLock<u64> = OnceLock::new();
    let salt = *SALT.get_or_init(|| RandomState::new().hash_one(0xC0FFEEu64));

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{nanos:x}-{salt:016x}-{seq:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn append_then_reopen_restores_transcript() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let id = store.meta().id.clone();

        let mut session = Session::new();
        session.user("hello");
        store.append_new(&session.messages).unwrap();
        session.user("world");
        store.append_new(&session.messages).unwrap();

        let (_store, restored) = repo.open_id(&id).unwrap();
        assert_eq!(restored.messages.len(), 2);
        let dump = serde_json::to_string(restored.messages.as_ref()).unwrap();
        assert!(dump.contains("hello") && dump.contains("world"));
    }

    #[test]
    fn torn_last_line_is_recovered() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("first");
        store.append_new(&session.messages).unwrap();

        // Simulate a crash mid-append: a half-written, unterminated JSON line.
        let path = repo.find_path(&id).unwrap();
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        write!(f, "{{\"type\":\"message\",\"role\":\"user\",\"cont").unwrap();
        drop(f);

        let (_store, restored) = repo.open_id(&id).unwrap();
        // The intact first message survives; the torn record is dropped.
        assert_eq!(restored.messages.len(), 1);
    }

    #[test]
    fn rewrite_replaces_whole_transcript() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("a");
        session.user("b");
        session.user("c");
        store.append_new(&session.messages).unwrap();

        // Compaction-style rewrite to a shorter list.
        let compacted = vec![Message::user("summary")];
        store.rewrite(&compacted).unwrap();

        let (_store, restored) = repo.open_id(&id).unwrap();
        assert_eq!(restored.messages.len(), 1);
        let dump = serde_json::to_string(restored.messages.as_ref()).unwrap();
        assert!(dump.contains("summary") && !dump.contains("\"a\""));
    }

    #[test]
    fn list_is_newest_first_and_fork_links_parent() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut a = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut sa = Session::new();
        sa.user("one");
        sa.user("two");
        a.append_new(&sa.messages).unwrap();
        let id_a = a.meta().id.clone();

        // Fork the first message only.
        let (forked, fsession) = repo.fork(&id_a, 1).unwrap();
        assert_eq!(fsession.messages.len(), 1);
        assert_eq!(forked.meta().parent.as_deref(), Some(id_a.as_str()));

        let metas = repo.list().unwrap();
        assert_eq!(metas.len(), 2);
        // Newest first: the fork was created after `a`.
        assert!(metas[0].created_at >= metas[1].created_at);
    }

    #[test]
    fn set_title_persists() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("hi");
        store.append_new(&session.messages).unwrap();
        store.set_title("My Session", &session.messages).unwrap();
        let metas = repo.list().unwrap();
        let found = metas.iter().find(|m| m.id == id).unwrap();
        assert_eq!(found.title.as_deref(), Some("My Session"));
        // A pure title rewrite drops no messages, so it leaves no compaction provenance.
        assert_eq!(found.compactions, 0);
    }

    #[test]
    fn delete_removes_session() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        assert_eq!(repo.list().unwrap().len(), 1);
        repo.delete(&id).unwrap();
        assert_eq!(repo.list().unwrap().len(), 0);
    }

    #[test]
    fn delete_is_idempotent() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();

        // First delete removes it; a second delete (and a delete of a never-seen id) are no-op successes.
        repo.delete(&id).unwrap();
        repo.delete(&id).unwrap();
        repo.delete("does-not-exist").unwrap();
        assert!(repo.list().unwrap().is_empty());
    }

    #[test]
    fn ids_are_unique_and_shaped() {
        // All ids in a batch are distinct (the per-process salt + monotonic seq guarantee it within a
        // process; the salt guards across processes), and the shape is `<hex>-<hex>-<hex>`.
        let ids: std::collections::HashSet<String> = (0..10_000).map(|_| new_id()).collect();
        assert_eq!(ids.len(), 10_000);

        let id = new_id();
        assert_eq!(id.matches('-').count(), 2);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
    }

    #[test]
    fn list_reports_updated_count_and_preview() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("hello world, this is the first message");
        session.user("second");
        store.append_new(&session.messages).unwrap();

        let listings = repo.list().unwrap();
        let l = listings.iter().find(|l| l.id == id).unwrap();
        assert_eq!(l.message_count, 2);
        assert_eq!(
            l.preview.as_deref(),
            Some("hello world, this is the first message")
        );
        assert!(l.updated_at > 0);
    }

    #[test]
    fn preview_truncates_long_first_message() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("x".repeat(PREVIEW_MAX + 50));
        store.append_new(&session.messages).unwrap();

        let listings = repo.list().unwrap();
        let preview = listings[0].preview.as_deref().unwrap();
        // PREVIEW_MAX chars plus the ellipsis marker.
        assert_eq!(preview.chars().count(), PREVIEW_MAX + 1);
        assert!(preview.ends_with('…'));
    }

    #[test]
    fn newer_version_is_rejected() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let path = repo.find_path(&id).unwrap();

        // Rewrite the header with a version newer than this build understands.
        let content = fs::read_to_string(&path).unwrap();
        let mut lines = content.lines();
        let mut header: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        header["version"] = serde_json::json!(999);
        let rest: Vec<&str> = lines.collect();
        fs::write(&path, format!("{}\n{}", header, rest.join("\n"))).unwrap();

        // Opening errors clearly rather than mis-parsing... (`map` drops the non-`Debug` success value
        // so `unwrap_err` can format it).
        let err = repo.open_id(&id).map(|_| ()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        // ...and the listing skips the unreadable file instead of including it.
        assert!(repo.list().unwrap().is_empty());
    }

    #[test]
    fn writes_round_trip_after_fsync() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("durable");
        // Exercises the synced append path...
        store.append_new(&session.messages).unwrap();
        // ...and the synced atomic-rewrite path.
        store.rewrite(&session.messages).unwrap();

        let (_store, restored) = repo.open_id(&id).unwrap();
        assert_eq!(restored.messages.len(), 1);
        let dump = serde_json::to_string(restored.messages.as_ref()).unwrap();
        assert!(dump.contains("durable"));
    }

    #[test]
    fn compaction_records_provenance() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("a");
        session.user("b");
        session.user("c");
        store.append_new(&session.messages).unwrap();

        // Compact 3 messages down to 1 — two dropped.
        store.rewrite(&[Message::user("summary")]).unwrap();
        assert_eq!(store.meta().compactions, 1);
        assert_eq!(store.meta().dropped_messages, 2);

        // Provenance is persisted in the header and survives reopen.
        let (reopened, _) = repo.open_id(&id).unwrap();
        assert_eq!(reopened.meta().compactions, 1);
        assert_eq!(reopened.meta().dropped_messages, 2);
    }
}
