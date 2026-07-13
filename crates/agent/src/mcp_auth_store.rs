//! Persisted MCP OAuth credentials (`~/.claude/mcp_auth.json`), keyed by MCP server name — the
//! credential-storage backend `agent mcp-login`/`agent mcp-logout`/`tools::mcp::connect_http` plug
//! into `rmcp`'s own [`CredentialStore`] trait so a login (and every later token refresh) persists
//! across process restarts, the same way `AuthStore` (`auth_store.rs`) does for model-provider OAuth.
//!
//! Same on-disk pattern as `auth_store.rs`/`trust_store.rs`/`settings.rs`: own small [`FileLock`],
//! atomic writes (`crate::tools::write_atomic`), re-read-under-lock-before-mutate so two concurrent
//! writers (an `agent mcp-login` for one server racing a token refresh for another, say) can't clobber
//! each other — deliberately not sharing code with any of the other three stores, matching their own
//! stated convention (each is small and self-contained; there's no shared abstraction worth extracting
//! for a handful of lines of file-locking boilerplate).

use std::collections::BTreeMap;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use rmcp::transport::auth::{AuthError, CredentialStore, StoredCredentials};

use crate::settings::config_dir_root;

/// On-disk shape: every configured MCP server's own stored credential, keyed by the server's `name`
/// in `mcp_servers` — matches how `tools::mcp` already identifies a server everywhere else (the
/// `mcp__<server>__<tool>` registration prefix). A server renamed in `settings.json` loses its stored
/// login (there's no way to know the old and new names refer to "the same" server) — an accepted,
/// narrow limitation, not a bug: re-running `agent mcp-login <new-name>` is a one-time cost.
#[derive(Default, serde::Serialize, serde::Deserialize)]
struct OnDisk(BTreeMap<String, StoredCredentials>);

/// A persisted MCP OAuth credential store, one per machine (`~/.claude/mcp_auth.json`).
pub struct McpAuthStore {
    path: PathBuf,
}

impl McpAuthStore {
    /// Open the default store. A missing or unparsable file is treated as "no server has ever logged
    /// in", never an error — matches every sibling store's identical tolerance.
    pub fn open_default() -> Self {
        Self {
            path: default_path(),
        }
    }

    /// A [`CredentialStore`] scoped to `server_name`'s own slot in this file — what an
    /// `AuthorizationManager` actually reads and writes through once
    /// [`AuthorizationManager::set_credential_store`](rmcp::transport::auth::AuthorizationManager::set_credential_store)
    /// is given one of these.
    pub fn scoped(&self, server_name: &str) -> ScopedMcpCredentialStore {
        ScopedMcpCredentialStore {
            path: self.path.clone(),
            server_name: server_name.to_string(),
        }
    }

    /// Whether `server_name` has a persisted credential at all — `tools::mcp::connect_http` checks
    /// this before ever attempting `initialize_from_store` (no point standing up an
    /// `AuthorizationManager`/doing a metadata-discovery round trip for a server nobody has logged
    /// into), and `agent mcp-login`/`mcp-logout` use it to report current status.
    pub fn has_credential(&self, server_name: &str) -> bool {
        read_store_file(&self.path).0.contains_key(server_name)
    }

    /// Remove `server_name`'s stored credential, if any — `agent mcp-logout`.
    pub fn clear(&self, server_name: &str) -> std::io::Result<()> {
        mutate_locked(&self.path, |store| {
            store.0.remove(server_name);
        })
    }
}

/// One MCP server's own view of the shared store file — the [`CredentialStore`] impl an
/// `AuthorizationManager` for that server is configured with. Reads/writes only ever touch this
/// server's own `server_name` key; every other server's stored credential in the same file is
/// untouched.
pub struct ScopedMcpCredentialStore {
    path: PathBuf,
    server_name: String,
}

#[async_trait::async_trait]
impl CredentialStore for ScopedMcpCredentialStore {
    async fn load(&self) -> Result<Option<StoredCredentials>, AuthError> {
        Ok(read_store_file(&self.path)
            .0
            .get(&self.server_name)
            .cloned())
    }

    async fn save(&self, credentials: StoredCredentials) -> Result<(), AuthError> {
        mutate_locked(&self.path, |store| {
            store.0.insert(self.server_name.clone(), credentials);
        })
        .map_err(|e| AuthError::InternalError(e.to_string()))
    }

    async fn clear(&self) -> Result<(), AuthError> {
        mutate_locked(&self.path, |store| {
            store.0.remove(&self.server_name);
        })
        .map_err(|e| AuthError::InternalError(e.to_string()))
    }
}

fn read_store_file(path: &Path) -> OnDisk {
    let contents = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == ErrorKind::NotFound => return OnDisk::default(),
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "could not read mcp auth store, treating it as empty"
            );
            return OnDisk::default();
        }
    };
    match serde_json::from_str(&contents) {
        Ok(store) => store,
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "mcp auth store is not valid JSON, treating it as empty"
            );
            OnDisk::default()
        }
    }
}

fn write_store_file(path: &Path, store: &OnDisk) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    // This file holds live MCP OAuth access/refresh tokens. `write_atomic` only *preserves* an
    // existing file's permission bits — it never imposes one on a brand-new file, so the very first
    // write would otherwise land at the umask default (commonly world-readable). Create it `0600`
    // atomically first, exactly as `auth_store.rs` does for `auth.json`.
    if !path.exists() {
        create_private(path)?;
    }
    let body = serde_json::to_string_pretty(store).map_err(std::io::Error::other)?;
    let path_str = path
        .to_str()
        .ok_or_else(|| std::io::Error::other(format!("non-UTF-8 path: {}", path.display())))?;
    crate::tools::write_atomic(path_str, body.as_bytes())?;
    // Belt-and-suspenders on top of the atomic-at-creation `0600`: `write_atomic` only ever copies an
    // existing file's mode forward, never re-asserts one, so a file loosened out-of-band (a stray
    // `chmod`, a restore from backup) would stay that way forever. Re-assert `0600` after every write
    // so it self-heals — mirrors `auth_store.rs::set_private`.
    set_private(path)
}

/// Create an empty file at `path` with `0600` permissions set atomically at creation (Unix), closing
/// the window a `set_permissions`-afterward approach would leave open. Mirrors
/// `auth_store.rs`/`session_store.rs`'s identical helper (duplicated, not shared, per this codebase's
/// small-self-contained-stores convention).
fn create_private(path: &Path) -> std::io::Result<()> {
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path)?;
    Ok(())
}

/// Unconditionally re-assert `0600` on an already-written file (Unix only — a no-op elsewhere), so a
/// mode loosened out-of-band self-heals on the next write. Mirrors `auth_store.rs::set_private`.
fn set_private(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

/// Acquire the cross-process lock, re-read the store's current on-disk state, apply `mutate`, persist
/// it — see `settings.rs::SettingsStore::mutate_locked`'s identical reasoning (a login for one server
/// and a token refresh for another must not race and silently clobber each other).
fn mutate_locked(path: &Path, mutate: impl FnOnce(&mut OnDisk)) -> std::io::Result<()> {
    let _lock = FileLock::acquire(path)?;
    let mut store = read_store_file(path);
    mutate(&mut store);
    write_store_file(path, &store)
}

const LOCK_TIMEOUT: Duration = Duration::from_secs(5);
const LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(20);
const STALE_LOCK_AGE: Duration = Duration::from_secs(10);

/// A cross-process advisory lock via atomic lockfile creation — see
/// `trust_store.rs::FileLock`/`settings.rs::FileLock`'s identical pattern, duplicated here rather than
/// shared (see this module's own doc comment for why).
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
                        let _ = fs::remove_file(&lock_path);
                        continue;
                    }
                    if Instant::now() >= deadline {
                        return Err(std::io::Error::new(
                            ErrorKind::TimedOut,
                            format!(
                                "timed out waiting for mcp auth store lock at {}",
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
    config_dir_root().join("mcp_auth.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_at(dir: &Path) -> McpAuthStore {
        McpAuthStore {
            path: dir.join("mcp_auth.json"),
        }
    }

    fn fake_credential(client_id: &str) -> StoredCredentials {
        StoredCredentials::new(client_id.to_string(), None, vec![], None)
    }

    #[tokio::test]
    async fn missing_store_file_has_no_credential_for_any_server() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_at(dir.path());
        assert!(!store.has_credential("fixture"));
        assert!(store.scoped("fixture").load().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn a_saved_credential_round_trips_and_is_visible_via_has_credential() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_at(dir.path());
        store
            .scoped("fixture")
            .save(fake_credential("client-1"))
            .await
            .unwrap();

        assert!(store.has_credential("fixture"));
        let loaded = store.scoped("fixture").load().await.unwrap().unwrap();
        assert_eq!(loaded.client_id, "client-1");
    }

    #[tokio::test]
    async fn credentials_are_isolated_per_server_name() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_at(dir.path());
        store
            .scoped("alpha")
            .save(fake_credential("alpha-client"))
            .await
            .unwrap();
        store
            .scoped("beta")
            .save(fake_credential("beta-client"))
            .await
            .unwrap();

        assert_eq!(
            store
                .scoped("alpha")
                .load()
                .await
                .unwrap()
                .unwrap()
                .client_id,
            "alpha-client"
        );
        assert_eq!(
            store
                .scoped("beta")
                .load()
                .await
                .unwrap()
                .unwrap()
                .client_id,
            "beta-client"
        );
        assert!(store.scoped("gamma").load().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn clear_removes_only_the_named_servers_credential() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_at(dir.path());
        store
            .scoped("alpha")
            .save(fake_credential("alpha-client"))
            .await
            .unwrap();
        store
            .scoped("beta")
            .save(fake_credential("beta-client"))
            .await
            .unwrap();

        store.clear("alpha").unwrap();

        assert!(!store.has_credential("alpha"));
        assert!(store.has_credential("beta"));
    }

    #[tokio::test]
    async fn scoped_clear_via_the_credential_store_trait_also_works() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_at(dir.path());
        let scoped = store.scoped("fixture");
        scoped.save(fake_credential("client-1")).await.unwrap();
        assert!(scoped.load().await.unwrap().is_some());

        scoped.clear().await.unwrap();
        assert!(scoped.load().await.unwrap().is_none());
    }

    #[test]
    fn a_corrupt_store_file_warns_and_is_treated_as_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp_auth.json");
        fs::write(&path, "not valid json { [ }").unwrap();

        let capture = crate::tracing_test::capture(|| {
            let store = read_store_file(&path);
            assert!(store.0.is_empty());
        });
        assert!(
            capture.messages().iter().any(|m| m.contains("mcp auth")),
            "a corrupt store file must be logged, not silently swallowed: {:?}",
            capture.messages()
        );
    }
}
