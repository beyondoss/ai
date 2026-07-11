//! [`FileBackend`] — the default memory store: local `*.md` files under a per-project directory.
//!
//! Layout: `~/.claude/projects/<encoded-cwd>/memory/`, reusing [`crate::settings::config_dir_root`]
//! (honors `AI_AGENT_CONFIG_DIR`) and [`crate::session_store::encode_cwd`] so a repo's memory sits
//! beside — and is scoped exactly like — its sessions. All worktrees of one repo share it.
//!
//! Durability follows this crate's established store discipline (`auth_store`/`trust_store`): every
//! mutation runs under a cross-process advisory [`FileLock`] and writes through
//! [`crate::tools::write_atomic`] (temp file + atomic rename), so a crash mid-write can't leave a
//! half-written document. Reads are resilient — a missing store is simply empty, an unreadable file
//! `warn!`s and is skipped rather than panicking. The [`FileLock`] is duplicated here rather than shared,
//! matching the convention documented in `auth_store.rs`: each store's lock is small and self-contained.

use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

use async_trait::async_trait;

use super::{
    Entry, Hit, INDEX_FILE, INDEX_MAX_BYTES, INDEX_MAX_LINES, MEMORY_ROOT, MemPath, MemoryBackend,
    MemoryError, SESSION_ROOT, View,
};

/// Where a [`FileBackend`]'s directory comes from. A durable store lives at a `Fixed` path for its whole
/// life; a session working-memory store points at a `Shared`, atomically-swappable path so a serve
/// session switch (`switch_session`/`new_session`/`fork`) re-points every holder — parent and shared
/// subagents alike — with one cell write, no tool or agent rebuild. See [`SessionDir`].
#[derive(Clone)]
enum DirSource {
    Fixed(PathBuf),
    Shared(SessionDir),
}

impl DirSource {
    /// The current directory. Cheap (a clone / a short read lock) — called once per operation.
    fn get(&self) -> PathBuf {
        match self {
            DirSource::Fixed(p) => p.clone(),
            // Recover a poisoned lock rather than panicking: a path cell has no invariant a panicked
            // writer could have left half-updated (it's a single atomic assignment), and the workspace
            // forbids `unwrap`. The worst case of a poisoned read is a stale-but-valid path.
            DirSource::Shared(cell) => cell.0.read().unwrap_or_else(|e| e.into_inner()).clone(),
        }
    }
}

/// A shared, swappable session-memory directory. Cloned (by `Arc`) into the session [`FileBackend`] and
/// held by the host; the host re-points it on a session switch and every backend clone sees the change.
#[derive(Clone)]
pub struct SessionDir(Arc<RwLock<PathBuf>>);

impl SessionDir {
    /// A new cell starting at `dir`.
    pub fn new(dir: PathBuf) -> Self {
        Self(Arc::new(RwLock::new(dir)))
    }

    /// Re-point the cell at `dir` — the next memory operation (parent or subagent) uses it.
    pub fn set(&self, dir: PathBuf) {
        *self.0.write().unwrap_or_else(|e| e.into_inner()) = dir;
    }
}

/// A memory store backed by a directory of files.
pub struct FileBackend {
    dir: DirSource,
    /// The logical root this store is surfaced under (e.g. [`MEMORY_ROOT`] or [`SESSION_ROOT`]). Only
    /// affects the paths reported back to the model (listings, search hits) — the on-disk layout is the
    /// same regardless — so one backend type serves either mount.
    root: &'static str,
}

impl FileBackend {
    /// The store for the project rooted at `cwd`: `~/.claude/projects/<encoded-cwd>/memory/`.
    pub fn for_project(cwd: &Path) -> Self {
        let canonical = crate::session_store::canonical_cwd(cwd);
        let encoded = crate::session_store::encode_cwd(&canonical.to_string_lossy());
        let dir = crate::settings::config_dir_root()
            .join("projects")
            .join(encoded)
            .join("memory");
        Self {
            dir: DirSource::Fixed(dir),
            root: MEMORY_ROOT,
        }
    }

    /// A durable store at an explicit directory (a `--memory <path>` / `file://` override).
    pub fn at(dir: PathBuf) -> Self {
        Self {
            dir: DirSource::Fixed(dir),
            root: MEMORY_ROOT,
        }
    }

    /// A session working-memory store at a fixed `dir`, surfaced under [`SESSION_ROOT`] (`/session`). For
    /// hosts with a single, non-switching session (`run`) and for tests.
    pub fn session_at(dir: PathBuf) -> Self {
        Self {
            dir: DirSource::Fixed(dir),
            root: SESSION_ROOT,
        }
    }

    /// A session working-memory store whose directory tracks a shared [`SessionDir`] cell — for a host
    /// (`serve`) that switches between sessions in one process.
    pub fn session_shared(cell: SessionDir) -> Self {
        Self {
            dir: DirSource::Shared(cell),
            root: SESSION_ROOT,
        }
    }

    /// The store's current base directory.
    fn dir(&self) -> PathBuf {
        self.dir.get()
    }

    /// The real filesystem path for a logical [`MemPath`].
    fn resolve(&self, path: &MemPath) -> PathBuf {
        if path.is_root() {
            self.dir()
        } else {
            self.dir().join(path.rel())
        }
    }

    /// The store-wide lock path guarding every mutation (one lock for the whole store keeps cross-file
    /// operations like `rename` and index updates consistent).
    fn lock_path(&self) -> PathBuf {
        self.dir().join(".memory.lock")
    }

    /// Acquire the store lock, having ensured the store directory exists.
    fn lock(&self) -> Result<FileLock, MemoryError> {
        fs::create_dir_all(self.dir()).map_err(|e| MemoryError::Backend(e.to_string()))?;
        FileLock::acquire(&self.lock_path()).map_err(|e| MemoryError::Backend(e.to_string()))
    }

    /// Read a document's text, distinguishing "no such file" ([`MemoryError::NotFound`]) from a real IO
    /// failure ([`MemoryError::Backend`]) and refusing a directory.
    fn read_doc(&self, path: &MemPath) -> Result<String, MemoryError> {
        let real = self.resolve(path);
        match fs::metadata(&real) {
            Ok(m) if m.is_dir() => Err(MemoryError::InvalidPath(format!(
                "{} is a directory, not a document",
                path.display()
            ))),
            Ok(_) => fs::read_to_string(&real).map_err(|e| MemoryError::Backend(e.to_string())),
            Err(e) if e.kind() == ErrorKind::NotFound => Err(MemoryError::NotFound(path.display())),
            Err(e) => Err(MemoryError::Backend(e.to_string())),
        }
    }

    /// Atomically write `text` to a document, creating parent directories. Assumes the caller holds the
    /// store lock.
    fn write_doc(&self, path: &MemPath, text: &str) -> Result<(), MemoryError> {
        let real = self.resolve(path);
        if let Some(parent) = real.parent() {
            fs::create_dir_all(parent).map_err(|e| MemoryError::Backend(e.to_string()))?;
        }
        let real_str = real
            .to_str()
            .ok_or_else(|| MemoryError::Backend(format!("non-UTF-8 path: {}", real.display())))?;
        crate::tools::write_atomic(real_str, text.as_bytes())
            .map_err(|e| MemoryError::Backend(e.to_string()))
    }

    /// Recursively list every document (and sub-directory) beneath `start`, as logical entries sorted by
    /// path. Skips the lock file and dotfiles. A missing directory yields an empty listing.
    fn listing(&self, start: &Path) -> Result<Vec<Entry>, MemoryError> {
        let mut out = Vec::new();
        self.walk(start, &mut out)?;
        out.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(out)
    }

    fn walk(&self, dir: &Path, out: &mut Vec<Entry>) -> Result<(), MemoryError> {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(MemoryError::Backend(e.to_string())),
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            // Never surface the lock file or any hidden bookkeeping to the model.
            if name.starts_with('.') {
                continue;
            }
            let real = entry.path();
            let Ok(logical) = real.strip_prefix(self.dir()) else {
                continue;
            };
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            let size = if is_dir {
                0
            } else {
                entry.metadata().map(|m| m.len()).unwrap_or(0)
            };
            out.push(Entry {
                path: format!("{}/{}", self.root, logical.to_string_lossy()),
                is_dir,
                size,
            });
            if is_dir {
                self.walk(&real, out)?;
            }
        }
        Ok(())
    }
}

/// Keep the first [`INDEX_MAX_LINES`] lines / [`INDEX_MAX_BYTES`] bytes of the index — whichever bites
/// first — so the always-injected prefix stays bounded.
fn cap_index(raw: &str) -> String {
    let mut out = String::new();
    for (i, line) in raw.lines().enumerate() {
        if i >= INDEX_MAX_LINES {
            out.push_str("\n[index truncated: showing the first ");
            out.push_str(&INDEX_MAX_LINES.to_string());
            out.push_str(" lines]");
            break;
        }
        if out.len() + line.len() + 1 > INDEX_MAX_BYTES {
            out.push_str("\n[index truncated at ~25 KB]");
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

#[async_trait]
impl MemoryBackend for FileBackend {
    async fn index(&self) -> Result<String, MemoryError> {
        let path = self.dir().join(INDEX_FILE);
        match fs::read_to_string(&path) {
            Ok(raw) => Ok(cap_index(&raw)),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(String::new()),
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "could not read MEMORY.md index, treating it as empty");
                Ok(String::new())
            }
        }
    }

    async fn view(
        &self,
        path: &MemPath,
        range: Option<(usize, usize)>,
    ) -> Result<View, MemoryError> {
        let real = self.resolve(path);
        let is_dir = path.is_root() || fs::metadata(&real).map(|m| m.is_dir()).unwrap_or(false);
        if is_dir {
            // A non-root path that doesn't exist at all is a NotFound, not an empty listing.
            if !path.is_root() && !real.exists() {
                return Err(MemoryError::NotFound(path.display()));
            }
            return Ok(View::Listing(self.listing(&real)?));
        }
        let text = self.read_doc(path)?;
        match range {
            None => Ok(View::Document(text)),
            Some((start, end)) => {
                // 1-indexed, inclusive, clamped — matching the text-editor `view_range` semantics.
                let start = start.max(1);
                let lines: Vec<&str> = text.lines().collect();
                if start > lines.len() {
                    return Ok(View::Document(String::new()));
                }
                let end = end.min(lines.len());
                let slice = if end >= start {
                    lines[start - 1..end].join("\n")
                } else {
                    String::new()
                };
                Ok(View::Document(slice))
            }
        }
    }

    async fn create(&self, path: &MemPath, text: &str) -> Result<(), MemoryError> {
        if path.is_root() {
            return Err(MemoryError::InvalidPath(
                "cannot create the memory root itself".to_string(),
            ));
        }
        let _lock = self.lock()?;
        let real = self.resolve(path);
        if real.is_dir() {
            return Err(MemoryError::InvalidPath(format!(
                "{} is a directory",
                path.display()
            )));
        }
        // Refuse to clobber existing durable knowledge: `create` makes a *new* document. To change one,
        // the model uses `str_replace`/`insert` (or `delete` then `create`) — a deliberate divergence
        // from `memory_20250818`'s overwrite-on-create, chosen because silently replacing a good memory
        // is exactly the failure a durable store must not have. The error tells the model what to do
        // instead, so it recovers on the next turn.
        if real.exists() {
            return Err(MemoryError::AlreadyExists(path.display()));
        }
        self.write_doc(path, text)
    }

    async fn str_replace(&self, path: &MemPath, old: &str, new: &str) -> Result<(), MemoryError> {
        let _lock = self.lock()?;
        // Re-read fresh under the lock, mutate, write back — the store discipline.
        let text = self.read_doc(path)?;
        let count = text.matches(old).count();
        if count != 1 {
            return Err(MemoryError::NotUnique {
                path: path.display(),
                old: old.to_string(),
                count,
            });
        }
        let replaced = text.replacen(old, new, 1);
        self.write_doc(path, &replaced)
    }

    async fn insert(&self, path: &MemPath, line: usize, text: &str) -> Result<(), MemoryError> {
        let _lock = self.lock()?;
        let existing = self.read_doc(path)?;
        let mut lines: Vec<&str> = existing.lines().collect();
        let at = line.min(lines.len());
        // Insert text (which may itself be multi-line) as its own lines after `at`.
        let inserted: Vec<&str> = text.split('\n').collect();
        for (offset, l) in inserted.into_iter().enumerate() {
            lines.insert(at + offset, l);
        }
        let mut joined = lines.join("\n");
        // Preserve a trailing newline if the original had one (or was empty and we appended content).
        if existing.ends_with('\n') || existing.is_empty() {
            joined.push('\n');
        }
        self.write_doc(path, &joined)
    }

    async fn delete(&self, path: &MemPath) -> Result<(), MemoryError> {
        if path.is_root() {
            return Err(MemoryError::InvalidPath(
                "cannot delete the memory root".to_string(),
            ));
        }
        let _lock = self.lock()?;
        let real = self.resolve(path);
        match fs::metadata(&real) {
            Ok(m) if m.is_dir() => {
                fs::remove_dir(&real).map_err(|e| {
                    // ENOTEMPTY: 39 on Linux, 66 on macOS/BSD. Checked via raw errno rather than
                    // `ErrorKind::DirectoryNotEmpty` to avoid depending on that variant's stabilization.
                    if e.raw_os_error() == Some(39) || e.raw_os_error() == Some(66) {
                        MemoryError::InvalidPath(format!(
                            "{} is a non-empty directory; delete its contents first",
                            path.display()
                        ))
                    } else {
                        MemoryError::Backend(e.to_string())
                    }
                })
            }
            Ok(_) => fs::remove_file(&real).map_err(|e| MemoryError::Backend(e.to_string())),
            Err(e) if e.kind() == ErrorKind::NotFound => Err(MemoryError::NotFound(path.display())),
            Err(e) => Err(MemoryError::Backend(e.to_string())),
        }
    }

    async fn rename(&self, from: &MemPath, to: &MemPath) -> Result<(), MemoryError> {
        if from.is_root() || to.is_root() {
            return Err(MemoryError::InvalidPath(
                "cannot rename the memory root".to_string(),
            ));
        }
        let _lock = self.lock()?;
        let src = self.resolve(from);
        let dst = self.resolve(to);
        if !src.exists() {
            return Err(MemoryError::NotFound(from.display()));
        }
        if dst.exists() {
            return Err(MemoryError::AlreadyExists(to.display()));
        }
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent).map_err(|e| MemoryError::Backend(e.to_string()))?;
        }
        fs::rename(&src, &dst).map_err(|e| MemoryError::Backend(e.to_string()))
    }

    async fn search(&self, query: &str) -> Result<Vec<Hit>, MemoryError> {
        let needle = query.to_lowercase();
        if needle.is_empty() {
            return Ok(Vec::new());
        }
        let dir = self.dir();
        let entries = self.listing(&dir)?;
        let mut hits = Vec::new();
        for entry in entries {
            if entry.is_dir {
                continue;
            }
            // Re-derive the real path from the logical one.
            let rel = entry
                .path
                .strip_prefix(&format!("{}/", self.root))
                .unwrap_or(&entry.path);
            let real = dir.join(rel);
            let Ok(text) = fs::read_to_string(&real) else {
                continue;
            };
            for (i, line) in text.lines().enumerate() {
                if line.to_lowercase().contains(&needle) {
                    hits.push(Hit {
                        path: entry.path.clone(),
                        line: i + 1,
                        text: line.to_string(),
                    });
                }
            }
        }
        Ok(hits)
    }
}

// ---- FileLock: a cross-process advisory lock, duplicated per this crate's store convention ----------

const LOCK_TIMEOUT: Duration = Duration::from_secs(5);
const LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(20);
const STALE_LOCK_AGE: Duration = Duration::from_secs(10);

/// Released by deleting the lock file on `Drop`, so a panicked or early-returning holder still frees it.
struct FileLock {
    path: PathBuf,
}

impl FileLock {
    fn acquire(lock_path: &Path) -> io::Result<Self> {
        if let Some(parent) = lock_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let deadline = Instant::now() + LOCK_TIMEOUT;
        loop {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(lock_path)
            {
                Ok(_) => {
                    return Ok(Self {
                        path: lock_path.to_path_buf(),
                    });
                }
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                    if is_stale(lock_path) {
                        let _ = fs::remove_file(lock_path);
                        continue;
                    }
                    if Instant::now() >= deadline {
                        return Err(io::Error::new(
                            ErrorKind::TimedOut,
                            format!(
                                "timed out waiting for memory store lock at {}",
                                lock_path.display()
                            ),
                        ));
                    }
                    std::thread::sleep(LOCK_RETRY_INTERVAL);
                }
                Err(e) => return Err(e),
            }
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn is_stale(lock_path: &Path) -> bool {
    fs::metadata(lock_path)
        .and_then(|m| m.modified())
        .and_then(|modified| {
            SystemTime::now()
                .duration_since(modified)
                .map_err(io::Error::other)
        })
        .is_ok_and(|age| age > STALE_LOCK_AGE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> (tempfile::TempDir, FileBackend) {
        let dir = tempfile::tempdir().unwrap();
        let b = FileBackend::at(dir.path().join("memory"));
        (dir, b)
    }

    fn p(s: &str) -> MemPath {
        MemPath::parse(s).unwrap()
    }

    #[tokio::test]
    async fn create_view_and_index_round_trip() {
        let (_d, b) = backend();
        b.create(&p("/memories/notes.md"), "hello\nworld\n")
            .await
            .unwrap();
        let View::Document(text) = b.view(&p("/memories/notes.md"), None).await.unwrap() else {
            panic!("expected a document");
        };
        assert_eq!(text, "hello\nworld\n");

        // A view of the root lists the store.
        let View::Listing(entries) = b.view(&p("/memories"), None).await.unwrap() else {
            panic!("expected a listing");
        };
        assert!(entries.iter().any(|e| e.path == "/memories/notes.md"));

        // The index reads MEMORY.md.
        assert_eq!(b.index().await.unwrap(), "");
        b.create(&p("/memories/MEMORY.md"), "- [notes](notes.md) — x\n")
            .await
            .unwrap();
        assert!(b.index().await.unwrap().contains("[notes]"));
    }

    #[tokio::test]
    async fn create_refuses_to_clobber_an_existing_memory() {
        // A durable store must not let `create` silently overwrite good knowledge — the model is told to
        // edit or delete first, and the original content is untouched.
        let (_d, b) = backend();
        b.create(&p("/memories/a.md"), "original").await.unwrap();
        let err = b.create(&p("/memories/a.md"), "clobber").await.unwrap_err();
        assert!(matches!(err, MemoryError::AlreadyExists(_)));
        let View::Document(t) = b.view(&p("/memories/a.md"), None).await.unwrap() else {
            panic!()
        };
        assert_eq!(
            t, "original",
            "a refused create must leave the file unchanged"
        );
    }

    #[tokio::test]
    async fn str_replace_requires_exactly_one_match() {
        let (_d, b) = backend();
        b.create(&p("/memories/a.md"), "foo bar foo\n")
            .await
            .unwrap();
        let err = b
            .str_replace(&p("/memories/a.md"), "foo", "baz")
            .await
            .unwrap_err();
        assert!(matches!(err, MemoryError::NotUnique { count: 2, .. }));

        b.str_replace(&p("/memories/a.md"), "bar", "BAR")
            .await
            .unwrap();
        let View::Document(t) = b.view(&p("/memories/a.md"), None).await.unwrap() else {
            panic!()
        };
        assert_eq!(t, "foo BAR foo\n");

        let zero = b
            .str_replace(&p("/memories/a.md"), "nope", "x")
            .await
            .unwrap_err();
        assert!(matches!(zero, MemoryError::NotUnique { count: 0, .. }));
    }

    #[tokio::test]
    async fn insert_after_line() {
        let (_d, b) = backend();
        b.create(&p("/memories/a.md"), "one\ntwo\nthree\n")
            .await
            .unwrap();
        b.insert(&p("/memories/a.md"), 1, "inserted").await.unwrap();
        let View::Document(t) = b.view(&p("/memories/a.md"), None).await.unwrap() else {
            panic!()
        };
        assert_eq!(t, "one\ninserted\ntwo\nthree\n");
    }

    #[tokio::test]
    async fn view_range_slices_lines() {
        let (_d, b) = backend();
        b.create(&p("/memories/a.md"), "l1\nl2\nl3\nl4\n")
            .await
            .unwrap();
        let View::Document(t) = b.view(&p("/memories/a.md"), Some((2, 3))).await.unwrap() else {
            panic!()
        };
        assert_eq!(t, "l2\nl3");
    }

    #[tokio::test]
    async fn delete_and_rename() {
        let (_d, b) = backend();
        b.create(&p("/memories/a.md"), "x").await.unwrap();
        b.rename(&p("/memories/a.md"), &p("/memories/b.md"))
            .await
            .unwrap();
        assert!(matches!(
            b.view(&p("/memories/a.md"), None).await.unwrap_err(),
            MemoryError::NotFound(_)
        ));
        b.delete(&p("/memories/b.md")).await.unwrap();
        assert!(matches!(
            b.delete(&p("/memories/b.md")).await.unwrap_err(),
            MemoryError::NotFound(_)
        ));
    }

    #[tokio::test]
    async fn rename_onto_existing_is_refused() {
        let (_d, b) = backend();
        b.create(&p("/memories/a.md"), "x").await.unwrap();
        b.create(&p("/memories/b.md"), "y").await.unwrap();
        assert!(matches!(
            b.rename(&p("/memories/a.md"), &p("/memories/b.md"))
                .await
                .unwrap_err(),
            MemoryError::AlreadyExists(_)
        ));
    }

    #[tokio::test]
    async fn search_finds_matches_case_insensitively() {
        let (_d, b) = backend();
        b.create(&p("/memories/a.md"), "The Build Command is mise\n")
            .await
            .unwrap();
        b.create(&p("/memories/b.md"), "nothing here\n")
            .await
            .unwrap();
        let hits = b.search("build command").await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "/memories/a.md");
        assert_eq!(hits[0].line, 1);
    }

    #[tokio::test]
    async fn a_session_store_reports_paths_under_the_session_root() {
        let dir = tempfile::tempdir().unwrap();
        let b = FileBackend::session_at(dir.path().join("session"));
        b.create(
            &MemPath::parse_in("/session/facts.md", SESSION_ROOT).unwrap(),
            "port 5433\n",
        )
        .await
        .unwrap();
        // Listings and search hits carry /session, not /memories.
        let View::Listing(entries) = b
            .view(&MemPath::parse_in("/session", SESSION_ROOT).unwrap(), None)
            .await
            .unwrap()
        else {
            panic!("expected a listing");
        };
        assert!(entries.iter().any(|e| e.path == "/session/facts.md"));
        let hits = b.search("5433").await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "/session/facts.md");
    }

    #[tokio::test]
    async fn missing_store_reads_empty() {
        let (_d, b) = backend();
        assert_eq!(b.index().await.unwrap(), "");
        let View::Listing(entries) = b.view(&p("/memories"), None).await.unwrap() else {
            panic!()
        };
        assert!(entries.is_empty());
    }

    #[test]
    fn cap_index_bounds_lines() {
        let big: String = (0..500).map(|i| format!("line {i}\n")).collect();
        let capped = cap_index(&big);
        // The kept content is bounded to INDEX_MAX_LINES; the truncation marker adds a couple of lines.
        assert!(capped.lines().count() <= INDEX_MAX_LINES + 3);
        assert!(capped.lines().filter(|l| l.starts_with("line ")).count() <= INDEX_MAX_LINES);
        assert!(capped.contains("index truncated"));
    }
}
