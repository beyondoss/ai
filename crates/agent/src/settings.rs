//! Persisted defaults for `run`/`serve` flags (model, gateway URL, session directory) — pi's own
//! `SettingsManager`, scoped down to what this crate's headless CLI surface actually has a use for.
//! Consulted as the last fallback after an explicit `--flag` or its `AI_AGENT_*`/`AI_GATEWAY_URL`
//! environment variable, before this crate's own built-in default — so an operator doesn't have to
//! retype the same flag on every invocation, without needing to export a shell environment variable
//! either. Managed out-of-band via `agent settings` (mirrors `agent trust`/`agent untrust` managing
//! `trust_store.rs`'s allowlist the same way), not through any `run`/`serve` RPC surface — both are
//! one-shot/headless processes with no live session that would need to *change* a stored default mid-run
//! the way pi's long-lived TUI does.
//!
//! Deliberately narrower than pi's own settings file, which also covers TUI-only concerns (theme,
//! terminal rendering, keybindings, external editor integration, telemetry) that don't apply to a
//! headless binary with no interactive UI of its own.

use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

/// On-disk shape: every field optional and `#[serde(default)]`, so a file missing a field (or not
/// existing at all) degrades to "no stored default" rather than a parse error.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Settings {
    /// Used when neither `--model` nor `AI_AGENT_MODEL` is given — pi's `defaultModel`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    /// Used when neither `--gateway-url` nor `AI_GATEWAY_URL` is given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_gateway_url: Option<String>,
    /// Used when neither `--session-dir` nor `AI_AGENT_SESSION_DIR` is given — pi's `sessionDir`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_session_dir: Option<String>,
}

/// A persisted settings file, one per machine (not per-project — matches pi's own global settings tier;
/// this crate has no per-project settings tier to layer on top of it).
pub struct SettingsStore {
    path: PathBuf,
    settings: Settings,
}

impl SettingsStore {
    /// Open the default settings store (`~/.claude/settings.json`). A missing or unparsable file is
    /// treated as all-defaults-unset, never an error.
    pub fn open_default() -> Self {
        Self::open(default_path())
    }

    pub fn open(path: PathBuf) -> Self {
        Self {
            settings: read_store_file(&path),
            path,
        }
    }

    /// The currently stored settings (as of this handle's last read or write — see `mutate_locked` for
    /// why a write always refreshes it against the file's live on-disk state first).
    pub fn get(&self) -> &Settings {
        &self.settings
    }

    /// Set (`Some`) or clear (`None`) the stored default model, persisting atomically.
    pub fn set_default_model(&mut self, model: Option<String>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_model = model)
    }

    /// Set (`Some`) or clear (`None`) the stored default gateway URL, persisting atomically.
    pub fn set_default_gateway_url(&mut self, url: Option<String>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_gateway_url = url)
    }

    /// Set (`Some`) or clear (`None`) the stored default session directory, persisting atomically.
    pub fn set_default_session_dir(&mut self, dir: Option<String>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_session_dir = dir)
    }

    /// Acquire the cross-process lock (see [`FileLock`]), re-read the store's *current* on-disk state —
    /// not `self`'s possibly-stale in-memory copy, which another process may have moved past since this
    /// handle's `open()` or its own last mutation — apply `mutate` to that fresh state, and persist it.
    /// `self` is updated to match afterward, so this handle's own `get()` stays consistent with what's
    /// now on disk. Mirrors `trust_store.rs::TrustStore::mutate_locked`'s identical reasoning: two
    /// `agent settings` invocations changing different fields concurrently must not let whichever writes
    /// last silently clobber the other's change.
    fn mutate_locked(&mut self, mutate: impl FnOnce(&mut Settings)) -> std::io::Result<()> {
        let _lock = FileLock::acquire(&self.path)?;
        let mut settings = read_store_file(&self.path);
        mutate(&mut settings);
        write_store_file(&self.path, &settings)?;
        self.settings = settings;
        Ok(())
    }
}

/// Read and parse the store file at `path`, tolerating a missing or unparsable file (all-unset) — shared
/// by [`SettingsStore::open`] and [`SettingsStore::mutate_locked`], which both need "current on-disk
/// state, gracefully defaulted."
fn read_store_file(path: &Path) -> Settings {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn write_store_file(path: &Path, settings: &Settings) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_string_pretty(settings).map_err(std::io::Error::other)?;
    let path_str = path
        .to_str()
        .ok_or_else(|| std::io::Error::other(format!("non-UTF-8 path: {}", path.display())))?;
    crate::tools::write_atomic(path_str, body.as_bytes())
}

/// How long [`FileLock::acquire`] retries before giving up.
const LOCK_TIMEOUT: Duration = Duration::from_secs(5);
/// How long between retries while waiting for another process's lock.
const LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(20);
/// A lock file older than this is assumed abandoned (its owner crashed without cleaning up, rather than
/// genuinely being mid-operation for this long — a settings read/parse/write is a handful of
/// milliseconds) and is forcibly reclaimed instead of blocking every future `agent settings` indefinitely.
const STALE_LOCK_AGE: Duration = Duration::from_secs(10);

/// A cross-process advisory lock via atomic lockfile creation — the same pattern
/// `trust_store.rs::FileLock` already uses for its own on-disk store, duplicated here rather than shared:
/// each store is small and self-contained, and the two have no other reason to depend on each other.
/// Released by deleting the lock file on `Drop`, so a panicked or early-returning holder still releases
/// it.
struct FileLock {
    path: PathBuf,
}

impl FileLock {
    fn acquire(store_path: &Path) -> std::io::Result<Self> {
        let lock_path = lock_path_for(store_path);
        if let Some(parent) = lock_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let deadline = Instant::now() + LOCK_TIMEOUT;
        loop {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(_) => return Ok(Self { path: lock_path }),
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                    if is_stale(&lock_path) {
                        // Best-effort: another process reclaiming it at the same instant just means our
                        // own retry loop's next `create_new` fails normally and we wait it out.
                        let _ = fs::remove_file(&lock_path);
                        continue;
                    }
                    if Instant::now() >= deadline {
                        return Err(std::io::Error::new(
                            ErrorKind::TimedOut,
                            format!(
                                "timed out waiting for settings store lock at {}",
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
                .map_err(std::io::Error::other)
        })
        .is_ok_and(|age| age > STALE_LOCK_AGE)
}

fn lock_path_for(store_path: &Path) -> PathBuf {
    let mut os = store_path.as_os_str().to_owned();
    os.push(".lock");
    PathBuf::from(os)
}

fn default_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join(".claude/settings.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_store_file_is_all_defaults_unset() {
        let dir = tempfile::tempdir().unwrap();
        let store = SettingsStore::open(dir.path().join("does-not-exist.json"));
        assert_eq!(store.get(), &Settings::default());
    }

    #[test]
    fn set_default_model_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        store
            .set_default_model(Some("claude-opus-4-8".to_string()))
            .unwrap();
        assert_eq!(
            store.get().default_model.as_deref(),
            Some("claude-opus-4-8")
        );

        let reopened = SettingsStore::open(path);
        assert_eq!(
            reopened.get().default_model.as_deref(),
            Some("claude-opus-4-8")
        );
    }

    #[test]
    fn clearing_a_field_removes_it_without_touching_others() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        store
            .set_default_model(Some("claude-opus-4-8".to_string()))
            .unwrap();
        store
            .set_default_gateway_url(Some("http://gw.internal".to_string()))
            .unwrap();

        store.set_default_model(None).unwrap();
        assert_eq!(store.get().default_model, None);
        assert_eq!(
            store.get().default_gateway_url.as_deref(),
            Some("http://gw.internal"),
            "clearing one field must not clobber another already-set field"
        );

        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().default_model, None);
        assert_eq!(
            reopened.get().default_gateway_url.as_deref(),
            Some("http://gw.internal")
        );
    }

    #[test]
    fn a_stale_in_memory_snapshot_is_refreshed_before_mutating_not_blindly_overwritten() {
        // Two handles on the same file (matching what actually happens across two separate `agent
        // settings` invocations) each setting a *different* field must not let one silently lose the
        // other's write — `handle_b` is opened before `handle_a`'s write, so its in-memory state is
        // genuinely stale by the time it mutates.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");

        let mut handle_a = SettingsStore::open(path.clone());
        let mut handle_b = SettingsStore::open(path.clone());

        handle_a
            .set_default_model(Some("model-a".to_string()))
            .unwrap();
        handle_b
            .set_default_session_dir(Some("/dir-b".to_string()))
            .unwrap();

        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().default_model.as_deref(), Some("model-a"));
        assert_eq!(
            reopened.get().default_session_dir.as_deref(),
            Some("/dir-b")
        );
    }

    #[test]
    fn store_creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        // A nested store path whose parent directory doesn't exist yet — matches the real
        // `~/.claude/settings.json` layout on a fresh `~/.claude`.
        let path = dir.path().join("nested/.claude/settings.json");
        let mut store = SettingsStore::open(path.clone());
        store.set_default_model(Some("m".to_string())).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn a_stale_lock_file_is_reclaimed_rather_than_blocking_forever() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let lock_path = lock_path_for(&path);
        fs::write(&lock_path, b"").unwrap();
        let stale = SystemTime::now() - (STALE_LOCK_AGE + Duration::from_secs(1));
        fs::File::open(&lock_path)
            .unwrap()
            .set_modified(stale)
            .unwrap();

        let mut store = SettingsStore::open(path);
        store.set_default_model(Some("m".to_string())).unwrap();
        assert_eq!(store.get().default_model.as_deref(), Some("m"));
    }

    #[test]
    fn unrecognized_fields_in_the_file_are_ignored_not_a_parse_error() {
        // Forward-compatibility: a settings file written by a future version with extra fields must
        // still open cleanly here, not fail outright.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(
            &path,
            r#"{"default_model":"m","theme":"dark","enableAnalytics":true}"#,
        )
        .unwrap();
        let store = SettingsStore::open(path);
        assert_eq!(store.get().default_model.as_deref(), Some("m"));
    }
}
