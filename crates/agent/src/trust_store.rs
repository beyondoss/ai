//! Project trust: gates the `SYSTEM.md` project override (and, per-project, skill/prompt-template
//! discovery) on an explicit allowlist.
//!
//! `resources::system_prompt_override` lets a project pin its own agent identity via
//! `<cwd>/.claude/SYSTEM.md` — but unconditionally honoring that let any repo checkout hijack the
//! agent's base instructions just by shipping the file. pi gates the same override (and its whole
//! project-local config bundle: skills, prompt templates, extensions) on an interactive "trust this
//! folder?" TUI prompt; this agent is headless (one-shot `run`, or `serve` driven by a client over
//! stdio) with no human at a terminal to ask, so trust here is either operator-asserted up front
//! (`--trust-project` / a `serve` init param) or recorded ahead of time in a persisted allowlist an
//! operator manages out-of-band (`agent trust <path>` / `agent untrust <path>`) — never granted
//! through an interactive prompt.
//!
//! Trust is tri-state and ancestor-aware: trusting `~/code` implies `~/code/anything` is trusted too,
//! unless a subdirectory was explicitly `untrust`ed, which shadows the inherited grant for that
//! subtree without touching the parent's entry.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The trust decision for a directory, after walking up to the nearest ancestor with an explicit
/// entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trust {
    Trusted,
    Untrusted,
    Unknown,
}

/// A persisted, tri-state, ancestor-aware allowlist of project paths.
pub struct TrustStore {
    path: PathBuf,
    trusted: BTreeSet<String>,
    untrusted: BTreeSet<String>,
}

/// On-disk shape once the store has been written by this version. Older files are a bare
/// `Vec<String>` (trusted-only) — see [`TrustStore::open`].
#[derive(Serialize, Deserialize, Default)]
struct StoreFile {
    #[serde(default)]
    trusted: BTreeSet<String>,
    #[serde(default)]
    untrusted: BTreeSet<String>,
}

impl TrustStore {
    /// Open the default trust store (`~/.claude/trusted-projects.json`). A missing or unparsable file
    /// is treated as an empty allowlist — nothing is trusted until explicitly recorded.
    pub fn open_default() -> Self {
        Self::open(default_path())
    }

    /// Open the trust store at `path`. Tries the current tri-state object shape first, falling back
    /// to the legacy bare `Vec<String>` shape (trusted-only) so files written by an older binary keep
    /// working with no migration step — the new shape is written the next time `trust`/`distrust` is
    /// called.
    pub fn open(path: PathBuf) -> Self {
        let contents = fs::read_to_string(&path).ok();
        let (trusted, untrusted) = contents
            .as_deref()
            .and_then(|s| serde_json::from_str::<StoreFile>(s).ok())
            .map(|f| (f.trusted, f.untrusted))
            .or_else(|| {
                contents
                    .as_deref()
                    .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
                    .map(|v| (v.into_iter().collect(), BTreeSet::new()))
            })
            .unwrap_or_default();
        Self {
            path,
            trusted,
            untrusted,
        }
    }

    /// The trust decision for `dir`, walking from `dir` up through its ancestors and returning the
    /// first explicit entry found (checking `untrusted` before `trusted` at each level, so a
    /// same-level conflict favors denying). No entry anywhere in the chain means [`Trust::Unknown`].
    pub fn lookup(&self, dir: &Path) -> Trust {
        let canonical = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        for ancestor in canonical.ancestors() {
            let key = ancestor.display().to_string();
            if self.untrusted.contains(&key) {
                return Trust::Untrusted;
            }
            if self.trusted.contains(&key) {
                return Trust::Trusted;
            }
        }
        Trust::Unknown
    }

    /// Whether `dir` is trusted (directly, or by inheriting an ancestor's trust grant).
    pub fn is_trusted(&self, dir: &Path) -> bool {
        matches!(self.lookup(dir), Trust::Trusted)
    }

    /// Idempotently trust `dir`, persisting atomically. Clears any exact-path `untrusted` entry for
    /// `dir` first, so re-trusting a directory that was previously explicitly distrusted actually
    /// takes effect rather than being shadowed by its own stale entry.
    pub fn trust(&mut self, dir: &Path) -> std::io::Result<()> {
        let key = canonical_key(dir);
        let removed_untrusted = self.untrusted.remove(&key);
        let inserted = self.trusted.insert(key);
        if !inserted && !removed_untrusted {
            return Ok(());
        }
        self.persist()
    }

    /// Idempotently mark `dir` as explicitly untrusted, persisting atomically. Clears any exact-path
    /// `trusted` entry for `dir` first (a directory can't be both).
    pub fn distrust(&mut self, dir: &Path) -> std::io::Result<()> {
        let key = canonical_key(dir);
        let removed_trusted = self.trusted.remove(&key);
        let inserted = self.untrusted.insert(key);
        if !inserted && !removed_trusted {
            return Ok(());
        }
        self.persist()
    }

    /// Remove any exact-path entry for `dir` — trusted *or* untrusted — without recording a new one,
    /// so `dir` reverts to inheriting whatever its nearest ancestor decides (or [`Trust::Unknown`] if
    /// none does), rather than staying pinned to its own explicit grant/denial. Neither `trust` nor
    /// `distrust` can express this: both always leave `dir` with *some* exact-path entry. Idempotent —
    /// clearing an already-unset directory is a no-op, not an error. Mirrors pi's own
    /// `ProjectTrustStore::setMany` accepting a `null` decision to delete an entry.
    pub fn clear(&mut self, dir: &Path) -> std::io::Result<()> {
        let key = canonical_key(dir);
        let removed_trusted = self.trusted.remove(&key);
        let removed_untrusted = self.untrusted.remove(&key);
        if !removed_trusted && !removed_untrusted {
            return Ok(());
        }
        self.persist()
    }

    fn persist(&self) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = StoreFile {
            trusted: self.trusted.clone(),
            untrusted: self.untrusted.clone(),
        };
        let body = serde_json::to_string_pretty(&file).map_err(std::io::Error::other)?;
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
        assert_eq!(store.lookup(dir.path()), Trust::Unknown);
    }

    #[test]
    fn trust_then_is_trusted() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        fs::create_dir_all(&project).unwrap();
        let store_path = dir.path().join("trusted-projects.json");

        let mut store = TrustStore::open(store_path.clone());
        assert!(!store.is_trusted(&project));
        store.trust(&project).unwrap();
        assert!(store.is_trusted(&project));

        // Persisted: a fresh handle on the same file sees the same trust.
        let reopened = TrustStore::open(store_path);
        assert!(reopened.is_trusted(&project));
    }

    #[test]
    fn trust_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        fs::create_dir_all(&project).unwrap();
        let store_path = dir.path().join("trusted-projects.json");

        let mut store = TrustStore::open(store_path.clone());
        store.trust(&project).unwrap();
        store.trust(&project).unwrap(); // must not error or duplicate

        let body = fs::read_to_string(&store_path).unwrap();
        let file: StoreFile = serde_json::from_str(&body).unwrap();
        assert_eq!(file.trusted.len(), 1);
    }

    #[test]
    fn missing_store_file_is_an_empty_allowlist() {
        let dir = tempfile::tempdir().unwrap();
        let store = TrustStore::open(dir.path().join("does-not-exist.json"));
        assert!(!store.is_trusted(dir.path()));
    }

    #[test]
    fn trust_creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        fs::create_dir_all(&project).unwrap();
        // A nested store path whose parent directory doesn't exist yet — matches the real
        // `~/.claude/trusted-projects.json` layout on a fresh `~/.claude`.
        let store_path = dir.path().join("nested/.claude/trusted-projects.json");

        let mut store = TrustStore::open(store_path.clone());
        store.trust(&project).unwrap();
        assert!(store_path.exists());
    }

    #[test]
    fn trusting_a_parent_implies_an_untrusted_child_dir() {
        let dir = tempfile::tempdir().unwrap();
        let code = dir.path().join("code");
        let sub = code.join("sub");
        let deeper = sub.join("deeper");
        let other = dir.path().join("other");
        fs::create_dir_all(&deeper).unwrap();
        fs::create_dir_all(&other).unwrap();
        let store_path = dir.path().join("trusted-projects.json");

        let mut store = TrustStore::open(store_path);
        store.trust(&code).unwrap();
        assert!(
            store.is_trusted(&sub),
            "child of a trusted dir inherits trust"
        );
        assert!(store.is_trusted(&deeper), "grandchild inherits too");
        assert!(!store.is_trusted(&other), "an unrelated dir stays unknown");

        store.distrust(&sub).unwrap();
        assert_eq!(store.lookup(&code), Trust::Trusted);
        assert_eq!(
            store.lookup(&sub),
            Trust::Untrusted,
            "an explicit distrust shadows the inherited parent grant"
        );
        assert_eq!(
            store.lookup(&deeper),
            Trust::Untrusted,
            "the shadow applies to the whole distrusted subtree"
        );
    }

    #[test]
    fn legacy_bare_array_file_still_parses_as_trusted_only() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        fs::create_dir_all(&project).unwrap();
        let store_path = dir.path().join("trusted-projects.json");
        let legacy = serde_json::to_string(&vec![canonical_key(&project)]).unwrap();
        fs::write(&store_path, legacy).unwrap();

        let store = TrustStore::open(store_path);
        assert!(store.is_trusted(&project));
    }

    #[test]
    fn retrusting_a_distrusted_dir_clears_the_distrust() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        fs::create_dir_all(&project).unwrap();
        let store_path = dir.path().join("trusted-projects.json");

        let mut store = TrustStore::open(store_path);
        store.distrust(&project).unwrap();
        assert_eq!(store.lookup(&project), Trust::Untrusted);
        store.trust(&project).unwrap();
        assert_eq!(store.lookup(&project), Trust::Trusted);
    }

    #[test]
    fn clear_reverts_to_inheriting_from_the_nearest_ancestor() {
        // Neither `trust` nor `distrust` can express "no opinion, inherit from the parent" — both
        // always leave an exact-path entry behind. `clear` removes it, so a subdirectory explicitly
        // distrusted earlier goes back to following its trusted parent, rather than needing a
        // `trust()` call that would instead pin it with its *own* explicit grant.
        let dir = tempfile::tempdir().unwrap();
        let code = dir.path().join("code");
        let sub = code.join("sub");
        fs::create_dir_all(&sub).unwrap();
        let store_path = dir.path().join("trusted-projects.json");

        let mut store = TrustStore::open(store_path);
        store.trust(&code).unwrap();
        store.distrust(&sub).unwrap();
        assert_eq!(store.lookup(&sub), Trust::Untrusted);

        store.clear(&sub).unwrap();
        assert_eq!(
            store.lookup(&sub),
            Trust::Trusted,
            "clearing the shadow reveals the inherited parent grant"
        );

        // Clearing the parent's own entry drops it all the way to Unknown (no ancestor left to
        // inherit from).
        store.clear(&code).unwrap();
        assert_eq!(store.lookup(&code), Trust::Unknown);
        assert_eq!(store.lookup(&sub), Trust::Unknown);
    }

    #[test]
    fn clear_on_an_already_unset_directory_is_a_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        fs::create_dir_all(&project).unwrap();
        let store_path = dir.path().join("trusted-projects.json");

        let mut store = TrustStore::open(store_path.clone());
        store.clear(&project).unwrap();
        assert_eq!(store.lookup(&project), Trust::Unknown);
        assert!(
            !store_path.exists(),
            "a no-op clear must not create the store file"
        );
    }
}
