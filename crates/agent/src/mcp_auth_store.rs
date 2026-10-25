//! Persisted MCP OAuth credentials (`~/.beyond/mcp_auth.json`), keyed by MCP server name — the
//! credential-storage backend `agent mcp-login`/`agent mcp-logout`/`tools::mcp::connect_http` plug
//! into `rmcp`'s own [`CredentialStore`] trait so a login (and every later token refresh) persists
//! across process restarts, the same way `AuthStore` (`auth_store.rs`) does for model-provider OAuth.
//!
//! Same on-disk pattern as `auth_store.rs`/`trust_store.rs`/`settings.rs`: the kernel lock all four
//! share (`crate::file_lock::lock_store_file`), atomic writes (`crate::tools::write_atomic`),
//! re-read-under-lock-before-mutate so two concurrent writers (an `agent mcp-login` for one server
//! racing a token refresh for another, say) can't clobber each other.

use std::collections::BTreeMap;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use rmcp::transport::auth::{AuthError, CredentialStore, StoredCredentials};
use serde::Deserialize;

use crate::settings::config_dir_root;

/// On-disk shape: every configured MCP server's own stored credential, keyed by the server's `name`
/// in `mcp_servers` — matches how `tools::mcp` already identifies a server everywhere else (the
/// `mcp__<server>__<tool>` registration prefix). A server renamed in `settings.json` loses its stored
/// login (there's no way to know the old and new names refer to "the same" server) — an accepted,
/// narrow limitation, not a bug: re-running `agent mcp-login <new-name>` is a one-time cost.
#[derive(Default)]
struct OnDisk(BTreeMap<String, StoredCredentials>);

/// A persisted MCP OAuth credential store, one per machine (`~/.beyond/mcp_auth.json`).
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
        mutate_locked(&self.path, |raw| raw.remove(server_name).is_some())
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
        // `read_store_file` is synchronous `fs` I/O; keep it off the tokio worker (same fix
        // `auth_credential_source.rs` applies for `AuthStore`) so it can't stall the reactor.
        let path = self.path.clone();
        let server_name = self.server_name.clone();
        crate::blocking::spawn(crate::blocking::Site::Store, move || {
            read_store_file(&path).0.get(&server_name).cloned()
        })
        .await
        .map_err(|e| AuthError::InternalError(e.to_string()))
    }

    async fn save(&self, credentials: StoredCredentials) -> Result<(), AuthError> {
        // `mutate_locked` holds a cross-process file lock and can `std::thread::sleep` up to
        // `LOCK_TIMEOUT` under contention — never on a tokio worker; hand it to `spawn_blocking`.
        let path = self.path.clone();
        let server_name = self.server_name.clone();
        crate::blocking::spawn(crate::blocking::Site::Store, move || {
            let value = serde_json::to_value(&credentials).map_err(std::io::Error::other)?;
            mutate_locked(&path, |raw| {
                raw.insert(server_name, value);
                true
            })
        })
        .await
        .map_err(|e| AuthError::InternalError(e.to_string()))?
        .map_err(|e| AuthError::InternalError(e.to_string()))
    }

    async fn clear(&self) -> Result<(), AuthError> {
        // Same blocking-under-lock concern as `save` — run off the reactor.
        let path = self.path.clone();
        let server_name = self.server_name.clone();
        crate::blocking::spawn(crate::blocking::Site::Store, move || {
            mutate_locked(&path, |raw| raw.remove(&server_name).is_some())
        })
        .await
        .map_err(|e| AuthError::InternalError(e.to_string()))?
        .map_err(|e| AuthError::InternalError(e.to_string()))
    }
}

/// Read the store for use: a missing, unreadable or corrupt file is empty (warned), and an entry that
/// doesn't parse as a credential is skipped on its own (warned) — one bad entry never hides every
/// other server's login. Never the basis of a write: see [`mutate_locked`].
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
    match serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&contents) {
        Ok(raw) => parse_entries(&raw),
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "mcp auth store is not a valid JSON object, treating it as empty"
            );
            OnDisk::default()
        }
    }
}

fn parse_entries(raw: &serde_json::Map<String, serde_json::Value>) -> OnDisk {
    let mut store = OnDisk::default();
    for (name, value) in raw {
        match StoredCredentials::deserialize(value) {
            Ok(credential) => {
                store.0.insert(name.clone(), credential);
            }
            Err(e) => {
                tracing::warn!(server = %name, error = %e, "could not parse a stored MCP credential, skipping");
            }
        }
    }
    store
}

/// Acquire the cross-process lock, re-read the store's current on-disk state, apply `mutate`, persist
/// it — see `settings.rs::SettingsStore::mutate_locked`'s identical reasoning (a login for one server
/// and a token refresh for another must not race and silently clobber each other).
///
/// The write follows [`crate::store_file`]'s rule: `mutate` edits the raw JSON object, so only the
/// server it names changes and every other entry — including one this binary can't parse — is
/// written back as it was; a file that can't be read or parsed refuses the write. The file holds
/// live tokens, so it is written owner-only.
fn mutate_locked(
    path: &Path,
    mutate: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>) -> bool,
) -> std::io::Result<()> {
    let _lock = crate::file_lock::lock_store_file(path)?;
    let mut raw = crate::store_file::read_object_for_write(path)?;
    if mutate(&mut raw) {
        crate::store_file::write_json(path, &serde_json::Value::Object(raw), true)?;
    }
    Ok(())
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

    /// What a writer that crashed while holding the lock leaves behind — the old protocol's
    /// `<file>.lock`, fresh, and an unheld lock file of the current one — does not block the next
    /// write. (The old lock made every write fail with `TimedOut` until that file aged out.)
    #[test]
    fn a_lock_left_by_a_crashed_writer_does_not_block_the_next_write() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_at(dir.path());
        fs::write(dir.path().join("mcp_auth.json.lock"), b"").unwrap();
        fs::write(dir.path().join("mcp_auth.json.beyond-lock"), b"").unwrap();
        let started = std::time::Instant::now();
        mutate_locked(&store.path, |s| {
            let value = serde_json::to_value(fake_credential("client-1")).unwrap();
            s.insert("fixture".to_string(), value);
            true
        })
        .unwrap();
        assert!(store.has_credential("fixture"));
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
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

/// A mutation writes back what it couldn't parse, and never overwrites a file it can't read.
#[cfg(test)]
mod roundtrip_tests {
    use super::*;
    use serde_json::{Value, json};

    #[tokio::test]
    async fn saving_one_server_keeps_an_entry_this_binary_cannot_parse() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp_auth.json");
        let foreign = json!({ "a-newer-shape": true });
        fs::write(&path, json!({ "other": foreign }).to_string()).unwrap();
        let store = McpAuthStore { path: path.clone() };
        store
            .scoped("mine")
            .save(StoredCredentials::new("c".into(), None, vec![], None))
            .await
            .unwrap();
        store.scoped("never-saved").clear().await.unwrap();
        let disk: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(disk["other"], foreign, "the unparseable entry must survive");
        assert!(store.has_credential("mine"));
    }

    #[tokio::test]
    async fn a_corrupt_file_is_left_byte_identical_and_the_mutation_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp_auth.json");
        let corrupt = b"{\"other\": {\"client_id\": ";
        fs::write(&path, corrupt).unwrap();
        let store = McpAuthStore { path: path.clone() };
        assert!(
            store
                .scoped("mine")
                .save(StoredCredentials::new("c".into(), None, vec![], None))
                .await
                .is_err()
        );
        assert!(store.clear("other").is_err());
        assert_eq!(fs::read(&path).unwrap(), corrupt);
    }
}
