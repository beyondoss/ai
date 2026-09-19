//! [`MapBackend`] — an in-process document map. Selected by `--memory memory://`.
//!
//! Nothing hits the filesystem. The store lives for the process (and is dropped with it), which is
//! what a test, a short-lived script, or an operator who wants the `memory` tool without a disk
//! path actually wants. Same document-map semantics as the Redis and Postgres backends — see
//! [`crate::memory::docs`].

use std::collections::BTreeMap;
use std::sync::RwLock;

use async_trait::async_trait;

use super::docs;
use super::{Hit, MEMORY_ROOT, MemPath, MemoryBackend, MemoryError, SESSION_ROOT, View};

/// In-process memory store. Cheap to construct; every mutation takes the write lock for the
/// duration of the document-map operation (one critical section, no partial writes).
pub struct MapBackend {
    docs: RwLock<BTreeMap<String, String>>,
    root: &'static str,
}

impl MapBackend {
    /// An empty durable store, surfaced under [`MEMORY_ROOT`].
    pub fn new() -> Self {
        Self {
            docs: RwLock::new(BTreeMap::new()),
            root: MEMORY_ROOT,
        }
    }

    /// An empty session store, surfaced under [`SESSION_ROOT`].
    pub fn session() -> Self {
        Self {
            docs: RwLock::new(BTreeMap::new()),
            root: SESSION_ROOT,
        }
    }

    fn read<T>(&self, f: impl FnOnce(&BTreeMap<String, String>) -> T) -> T {
        // A poisoned lock has no half-written map (every mutation is a single critical section
        // that either finishes or unwinds before the write is published). Recover the inner
        // map rather than panicking — the workspace forbids unwrap on a lock.
        f(&self.docs.read().unwrap_or_else(|e| e.into_inner()))
    }

    fn write<T>(
        &self,
        f: impl FnOnce(&mut BTreeMap<String, String>) -> Result<T, MemoryError>,
    ) -> Result<T, MemoryError> {
        f(&mut self.docs.write().unwrap_or_else(|e| e.into_inner()))
    }
}

impl Default for MapBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryBackend for MapBackend {
    async fn index(&self) -> Result<String, MemoryError> {
        Ok(self.read(docs::index))
    }

    async fn view(
        &self,
        path: &MemPath,
        range: Option<(usize, usize)>,
    ) -> Result<View, MemoryError> {
        self.read(|d| docs::view(d, path, range, self.root))
    }

    async fn create(&self, path: &MemPath, text: &str) -> Result<(), MemoryError> {
        self.write(|d| docs::create(d, path, text))
    }

    async fn str_replace(&self, path: &MemPath, old: &str, new: &str) -> Result<(), MemoryError> {
        self.write(|d| docs::str_replace(d, path, old, new))
    }

    async fn insert(&self, path: &MemPath, line: usize, text: &str) -> Result<(), MemoryError> {
        self.write(|d| docs::insert(d, path, line, text))
    }

    async fn delete(&self, path: &MemPath) -> Result<(), MemoryError> {
        self.write(|d| docs::delete(d, path))
    }

    async fn rename(&self, from: &MemPath, to: &MemPath) -> Result<(), MemoryError> {
        self.write(|d| docs::rename(d, from, to))
    }

    async fn search(&self, query: &str) -> Result<Vec<Hit>, MemoryError> {
        Ok(self.read(|d| docs::search(d, query, self.root)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> MemPath {
        MemPath::parse(s).unwrap()
    }

    #[tokio::test]
    async fn contract_matches_the_file_backend() {
        let b = MapBackend::new();
        b.create(&p("/memories/notes.md"), "hello\nworld\n")
            .await
            .unwrap();
        let View::Document(text) = b.view(&p("/memories/notes.md"), None).await.unwrap() else {
            panic!("expected a document");
        };
        assert_eq!(text, "hello\nworld\n");

        let View::Listing(entries) = b.view(&p("/memories"), None).await.unwrap() else {
            panic!("expected a listing");
        };
        assert!(entries.iter().any(|e| e.path == "/memories/notes.md"));

        assert_eq!(b.index().await.unwrap(), "");
        b.create(&p("/memories/MEMORY.md"), "- [notes](notes.md) — x\n")
            .await
            .unwrap();
        assert!(b.index().await.unwrap().contains("[notes]"));

        let err = b
            .create(&p("/memories/notes.md"), "clobber")
            .await
            .unwrap_err();
        assert!(matches!(err, MemoryError::AlreadyExists(_)));

        b.create(&p("/memories/a.md"), "foo bar foo\n")
            .await
            .unwrap();
        assert!(matches!(
            b.str_replace(&p("/memories/a.md"), "foo", "baz")
                .await
                .unwrap_err(),
            MemoryError::NotUnique { count: 2, .. }
        ));
        b.str_replace(&p("/memories/a.md"), "bar", "BAR")
            .await
            .unwrap();
        let View::Document(t) = b.view(&p("/memories/a.md"), None).await.unwrap() else {
            panic!()
        };
        assert_eq!(t, "foo BAR foo\n");

        b.create(&p("/memories/ins.md"), "one\ntwo\nthree\n")
            .await
            .unwrap();
        b.insert(&p("/memories/ins.md"), 1, "inserted")
            .await
            .unwrap();
        let View::Document(t) = b.view(&p("/memories/ins.md"), None).await.unwrap() else {
            panic!()
        };
        assert_eq!(t, "one\ninserted\ntwo\nthree\n");

        b.rename(&p("/memories/notes.md"), &p("/memories/renamed.md"))
            .await
            .unwrap();
        assert!(matches!(
            b.view(&p("/memories/notes.md"), None).await.unwrap_err(),
            MemoryError::NotFound(_)
        ));
        b.delete(&p("/memories/renamed.md")).await.unwrap();

        b.create(&p("/memories/s.md"), "The Build Command is mise\n")
            .await
            .unwrap();
        let hits = b.search("build command").await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "/memories/s.md");
    }

    #[tokio::test]
    async fn a_session_store_reports_paths_under_the_session_root() {
        let b = MapBackend::session();
        let path = MemPath::parse_in("/session/facts.md", SESSION_ROOT).unwrap();
        b.create(&path, "port 5433\n").await.unwrap();
        let View::Listing(entries) = b
            .view(&MemPath::parse_in("/session", SESSION_ROOT).unwrap(), None)
            .await
            .unwrap()
        else {
            panic!("expected a listing");
        };
        assert!(entries.iter().any(|e| e.path == "/session/facts.md"));
        let hits = b.search("5433").await.unwrap();
        assert_eq!(hits[0].path, "/session/facts.md");
    }
}
