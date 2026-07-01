//! Project trust: gates the `SYSTEM.md` project override on an explicit allowlist.
//!
//! `resources::system_prompt_override` lets a project pin its own agent identity via
//! `<cwd>/.claude/SYSTEM.md` — but unconditionally honoring that let any repo checkout hijack the
//! agent's base instructions just by shipping the file. pi gates the same override on an interactive
//! "trust this folder?" TUI prompt; this agent is headless (one-shot `run`, or `serve` driven by a
//! client over stdio) with no human at a terminal to ask, so trust here is either operator-asserted up
//! front (`--trust-project` / a `serve` init param) or recorded ahead of time in a persisted allowlist
//! an operator manages out-of-band (`agent trust <path>`) — never granted through an interactive prompt.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// A persisted allowlist of trusted absolute project paths.
pub struct TrustStore {
    path: PathBuf,
    trusted: BTreeSet<String>,
}

impl TrustStore {
    /// Open the default trust store (`~/.claude/trusted-projects.json`). A missing or unparsable file
    /// is treated as an empty allowlist — nothing is trusted until explicitly recorded.
    pub fn open_default() -> Self {
        Self::open(default_path())
    }

    /// Open the trust store at `path`.
    pub fn open(path: PathBuf) -> Self {
        let trusted = fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
            .map(|v| v.into_iter().collect())
            .unwrap_or_default();
        Self { path, trusted }
    }

    /// Whether `dir` is in the allowlist. Compares canonicalized paths so `.`/`..`/symlinks in how a
    /// caller happened to spell the directory don't cause a false negative; a directory that doesn't
    /// exist (or can't be canonicalized) falls back to comparing the path as given.
    pub fn is_trusted(&self, dir: &Path) -> bool {
        self.trusted.contains(&canonical_key(dir))
    }

    /// Idempotently add `dir` to the allowlist, persisting atomically (temp file + rename — the same
    /// pattern `write`/`edit` use). A no-op, still `Ok`, if `dir` is already trusted — no duplicate
    /// entries, no redundant write.
    pub fn add(&mut self, dir: &Path) -> std::io::Result<()> {
        let key = canonical_key(dir);
        if !self.trusted.insert(key) {
            return Ok(());
        }
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let list: Vec<&String> = self.trusted.iter().collect();
        let body = serde_json::to_string_pretty(&list).map_err(std::io::Error::other)?;
        let path_str = self.path.to_str().ok_or_else(|| {
            std::io::Error::other(format!("non-UTF-8 path: {}", self.path.display()))
        })?;
        crate::tools::write_atomic(path_str, body.as_bytes())
    }
}

/// Canonicalize `path` for allowlist comparison, falling back to the path as given when it doesn't
/// exist yet or can't be resolved (canonicalization needs the path to exist; a project directory
/// always does by the time trust is checked, but degrading gracefully here is cheap insurance).
fn canonical_key(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

fn default_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join(".claude/trusted-projects.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untrusted_directory_is_not_trusted() {
        let dir = tempfile::tempdir().unwrap();
        let store = TrustStore::open(dir.path().join("trusted-projects.json"));
        assert!(!store.is_trusted(dir.path()));
    }

    #[test]
    fn add_then_is_trusted() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        fs::create_dir_all(&project).unwrap();
        let store_path = dir.path().join("trusted-projects.json");

        let mut store = TrustStore::open(store_path.clone());
        assert!(!store.is_trusted(&project));
        store.add(&project).unwrap();
        assert!(store.is_trusted(&project));

        // Persisted: a fresh handle on the same file sees the same trust.
        let reopened = TrustStore::open(store_path);
        assert!(reopened.is_trusted(&project));
    }

    #[test]
    fn add_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        fs::create_dir_all(&project).unwrap();
        let store_path = dir.path().join("trusted-projects.json");

        let mut store = TrustStore::open(store_path.clone());
        store.add(&project).unwrap();
        store.add(&project).unwrap(); // must not error or duplicate

        let body = fs::read_to_string(&store_path).unwrap();
        let list: Vec<String> = serde_json::from_str(&body).unwrap();
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn missing_store_file_is_an_empty_allowlist() {
        let dir = tempfile::tempdir().unwrap();
        let store = TrustStore::open(dir.path().join("does-not-exist.json"));
        assert!(!store.is_trusted(dir.path()));
    }

    #[test]
    fn add_creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        fs::create_dir_all(&project).unwrap();
        // A nested store path whose parent directory doesn't exist yet — matches the real
        // `~/.claude/trusted-projects.json` layout on a fresh `~/.claude`.
        let store_path = dir.path().join("nested/.claude/trusted-projects.json");

        let mut store = TrustStore::open(store_path.clone());
        store.add(&project).unwrap();
        assert!(store_path.exists());
    }
}
