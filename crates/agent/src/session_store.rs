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
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use agent_core::{Message, Session};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// On-disk format version. Bump when the entry shapes change incompatibly.
const VERSION: u32 = 1;

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
        f.flush()?;
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
        f.flush()?;
        self.persisted = messages.len();
        Ok(())
    }

    /// Rewrite the whole file atomically (header + every message). Used when the message list was
    /// rewritten rather than extended — e.g. after compaction replaced the prefix with a summary.
    pub fn rewrite(&mut self, messages: &[Message]) -> std::io::Result<()> {
        let tmp = self.path.with_extension("jsonl.tmp");
        let mut f = File::create(&tmp)?;
        write_line(&mut f, &Entry::Session(self.meta.clone()))?;
        for msg in messages {
            write_line(&mut f, &Entry::Message(msg.clone()))?;
        }
        f.flush()?;
        fs::rename(&tmp, &self.path)?;
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

    /// All sessions' metadata, newest first. Files that fail to read (or lack a header) are skipped.
    pub fn list(&self) -> std::io::Result<Vec<SessionMeta>> {
        let mut metas = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            if let Some(meta) = read_header(&path) {
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

    /// Delete a session by id.
    pub fn delete(&self, id: &str) -> std::io::Result<()> {
        let path = self.find_path(id).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, format!("no session {id}"))
        })?;
        fs::remove_file(path)
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

/// Read just the header line of a session file, if valid.
fn read_header(path: &Path) -> Option<SessionMeta> {
    let file = File::open(path).ok()?;
    let mut first = String::new();
    BufReader::new(file).read_line(&mut first).ok()?;
    match serde_json::from_str::<Entry>(first.trim()).ok()? {
        Entry::Session(m) => Some(m),
        _ => None,
    }
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

/// A short, time-ordered, process-unique session id.
fn new_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{nanos:x}{seq:x}")
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
}
