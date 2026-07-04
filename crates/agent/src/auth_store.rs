//! Local storage for OAuth/subscription credentials (`~/.claude/auth.json` by default) — the Rust
//! analog of pi's own `auth.json`, holding what `crate::oauth`'s login/refresh flows produce.
//!
//! Same shape family as [`crate::settings`]/[`crate::trust_store`]: a flat JSON file, a hand-rolled
//! cross-process [`FileLock`] duplicated (not shared) into this file — matching this codebase's own
//! established convention that each store is small and self-contained — and a `mutate_locked`
//! re-read-fresh-under-lock discipline so two concurrent `agent` processes (a real, common pattern
//! for this tool) never silently clobber each other's write.
//!
//! **Deliberate divergence from pi**: pi's own `auth.json` can also hold a plain `api_key` credential
//! per provider (an alternative to OAuth). This crate already has a working static-credential
//! mechanism — the `--key`/`AI_AGENT_KEY` flag — so a second, parallel place to store a static key
//! would be two mechanisms for one job. This store is OAuth-only by design, not an oversight.

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

use crate::oauth::anthropic::AnthropicCredential;
use crate::oauth::github_copilot::GithubCopilotCredential;
use crate::oauth::openai_codex::OpenaiCodexCredential;
use crate::oauth::{OAuthCredential, OAuthProviderId};

/// A stored credential plus refresh bookkeeping.
#[derive(Debug, Clone)]
pub struct StoredCredential {
    pub credential: OAuthCredential,
    /// Set when the most recently *attempted* refresh failed (network error, or the provider
    /// rejected the refresh token) — `credential` is left exactly as it was; this is the only
    /// on-disk signal that the still-physically-present token is no longer trustworthy. Cleared on
    /// the next successful refresh or a fresh `agent login`. Surfaced by `auth-status` as
    /// `needs_reauth`.
    pub last_refresh_error: Option<String>,
}

/// A secret value that never appears in a `Debug` line and is best-effort scrubbed from memory on
/// drop — but, unlike `agent_core::client::ApiKey` (never serialized) or `beyond-gateway::Secret`
/// (redacts on serialize), round-trips its real value through `serde`. This store's whole job is to
/// persist the real value to disk; a redacting `Serialize` would defeat that, and duplicating either
/// existing type doesn't fit either shape. Same duplication convention as `FileLock` below: a small,
/// self-contained type for the one field shape that needs it here.
#[derive(Clone)]
pub struct Secret(String);

impl Secret {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Secret(***)")
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl Serialize for Secret {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for Secret {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Ok(Secret(String::deserialize(deserializer)?))
    }
}

/// The on-disk shape for one provider's entry — a flat object keyed by provider id at the file's top
/// level (see [`write_store_file`]), each holding this shape, matching pi's own `auth.json`
/// convention (`{"type":"oauth","access":...,"refresh":...,"expires":...,...extra}`) closely enough
/// to be recognizable, though this is this tool's own separate file, not literal interop with pi's.
#[derive(Serialize, Deserialize)]
struct OnDiskCredential {
    #[serde(rename = "type")]
    kind: String,
    access: Secret,
    refresh: Secret,
    expires: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    enterprise_url: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    available_model_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    account_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_refresh_error: Option<String>,
}

fn credential_to_on_disk(credential: &OAuthCredential, last_refresh_error: Option<String>) -> OnDiskCredential {
    let mut disk = OnDiskCredential {
        kind: "oauth".to_string(),
        access: Secret::new(credential.access()),
        refresh: Secret::new(credential.refresh_token()),
        expires: credential.expires_at_ms(),
        enterprise_url: None,
        available_model_ids: Vec::new(),
        account_id: None,
        last_refresh_error,
    };
    match credential {
        OAuthCredential::Anthropic(_) => {}
        OAuthCredential::GithubCopilot(c) => {
            disk.enterprise_url = c.enterprise_url.clone();
            disk.available_model_ids = c.available_model_ids.clone();
        }
        OAuthCredential::OpenaiCodex(c) => {
            disk.account_id = Some(c.account_id.clone());
        }
    }
    disk
}

fn on_disk_to_stored(provider: OAuthProviderId, disk: OnDiskCredential) -> StoredCredential {
    let credential = match provider {
        OAuthProviderId::Anthropic => OAuthCredential::Anthropic(AnthropicCredential {
            access: disk.access.expose().to_string(),
            refresh: disk.refresh.expose().to_string(),
            expires_at_ms: disk.expires,
        }),
        OAuthProviderId::GithubCopilot => OAuthCredential::GithubCopilot(GithubCopilotCredential {
            access: disk.access.expose().to_string(),
            refresh: disk.refresh.expose().to_string(),
            expires_at_ms: disk.expires,
            enterprise_url: disk.enterprise_url,
            available_model_ids: disk.available_model_ids,
        }),
        OAuthProviderId::OpenaiCodex => OAuthCredential::OpenaiCodex(OpenaiCodexCredential {
            access: disk.access.expose().to_string(),
            refresh: disk.refresh.expose().to_string(),
            expires_at_ms: disk.expires,
            account_id: disk.account_id.unwrap_or_default(),
        }),
    };
    StoredCredential {
        credential,
        last_refresh_error: disk.last_refresh_error,
    }
}

/// Failures from [`AuthStore::refreshed`].
#[derive(Debug)]
pub enum AuthError {
    /// No credential stored for the requested provider.
    NotLoggedIn,
    /// The refresh attempt itself failed — network error, or the provider rejected the refresh
    /// token. The stored credential is preserved on disk (see [`StoredCredential::last_refresh_error`]),
    /// not deleted, so the user can retry or re-run `agent login`.
    RefreshFailed(String),
    Io(io::Error),
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthError::NotLoggedIn => write!(f, "not logged in"),
            AuthError::RefreshFailed(msg) => write!(f, "refresh failed: {msg}"),
            AuthError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for AuthError {}

impl From<io::Error> for AuthError {
    fn from(e: io::Error) -> Self {
        AuthError::Io(e)
    }
}

/// Credential storage backed by a JSON file, one entry per provider.
pub struct AuthStore {
    path: PathBuf,
    credentials: BTreeMap<String, StoredCredential>,
}

impl AuthStore {
    /// Open the default store (`~/.claude/auth.json`, or `$AI_AGENT_CONFIG_DIR/auth.json`).
    pub fn open_default() -> Self {
        Self::open(default_path())
    }

    /// Open the store at `path`. A missing or unparsable file is treated as empty — nothing is
    /// logged in until explicitly recorded.
    pub fn open(path: PathBuf) -> Self {
        let credentials = read_store_file(&path);
        Self { path, credentials }
    }

    pub fn get(&self, provider: &str) -> Option<&StoredCredential> {
        self.credentials.get(provider)
    }

    pub fn list(&self) -> impl Iterator<Item = (&str, &StoredCredential)> {
        self.credentials.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Store (or overwrite) `provider`'s credential — e.g. after a fresh `agent login`. Always
    /// clears any prior `last_refresh_error`.
    pub fn set(&mut self, provider: &str, credential: OAuthCredential) -> io::Result<()> {
        self.mutate_locked(|credentials| {
            credentials.insert(
                provider.to_string(),
                StoredCredential {
                    credential,
                    last_refresh_error: None,
                },
            );
            true
        })
    }

    /// Remove `provider`'s credential, if any. Idempotent — returns whether one was actually present.
    pub fn remove(&mut self, provider: &str) -> io::Result<bool> {
        let mut removed = false;
        self.mutate_locked(|credentials| {
            removed = credentials.remove(provider).is_some();
            removed
        })?;
        Ok(removed)
    }

    /// The refresh-with-lock seam: acquire the cross-process lock, re-read the file **fresh from
    /// disk** (not `self`, which may be stale relative to another process), and only call `refresh`
    /// — while still holding the lock — if the credential is actually still expired by the time we
    /// look. This is the double-checked-locking property that makes concurrent `agent run`/`agent
    /// serve` processes sharing one OAuth credential safe: if the lock were released before the
    /// network refresh call, two processes could both decide "refresh needed" and both attempt to
    /// consume a single-use/rotating refresh token, and the second call would fail outright.
    ///
    /// On a refresh failure, the stored credential's `access`/`refresh`/`expires` are left bit-for-
    /// bit unchanged — only `last_refresh_error` is stamped — so a transient failure never loses a
    /// credential the user could still retry or re-login against.
    pub fn refreshed(
        &mut self,
        provider: &str,
        refresh: impl FnOnce(&OAuthCredential) -> std::result::Result<OAuthCredential, String>,
    ) -> std::result::Result<OAuthCredential, AuthError> {
        let _lock = FileLock::acquire(&self.path)?;
        let mut credentials = read_store_file(&self.path);
        let Some(stored) = credentials.get(provider) else {
            self.credentials = credentials;
            return Err(AuthError::NotLoggedIn);
        };

        if !stored.credential.is_expired(now_ms()) {
            // Another process (or an earlier caller) already refreshed since our last read.
            let current = stored.credential.clone();
            self.credentials = credentials;
            return Ok(current);
        }

        let current = stored.credential.clone();
        match refresh(&current) {
            Ok(fresh) => {
                credentials.insert(
                    provider.to_string(),
                    StoredCredential {
                        credential: fresh.clone(),
                        last_refresh_error: None,
                    },
                );
                write_store_file(&self.path, &credentials)?;
                self.credentials = credentials;
                Ok(fresh)
            }
            Err(message) => {
                if let Some(existing) = credentials.get_mut(provider) {
                    existing.last_refresh_error = Some(message.clone());
                }
                write_store_file(&self.path, &credentials)?;
                self.credentials = credentials;
                Err(AuthError::RefreshFailed(message))
            }
        }
    }

    /// Acquire the lock, re-read fresh, apply `mutate`, and persist only if it reports a change —
    /// same discipline as `TrustStore`/`SettingsStore`'s own `mutate_locked`.
    fn mutate_locked(
        &mut self,
        mutate: impl FnOnce(&mut BTreeMap<String, StoredCredential>) -> bool,
    ) -> io::Result<()> {
        let _lock = FileLock::acquire(&self.path)?;
        let mut credentials = read_store_file(&self.path);
        let changed = mutate(&mut credentials);
        if changed {
            write_store_file(&self.path, &credentials)?;
        }
        self.credentials = credentials;
        Ok(())
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Read and parse the store file at `path`, tolerating a missing file (empty store — nothing logged
/// in yet), an unreadable/corrupt file (`warn!`s rather than silently discarding), an unrecognized
/// provider id, or a per-entry parse failure (each entry is decoded independently, so one bad entry
/// doesn't take down every other stored credential).
fn read_store_file(path: &Path) -> BTreeMap<String, StoredCredential> {
    let contents = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == ErrorKind::NotFound => return BTreeMap::new(),
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "could not read auth store file, treating it as empty"
            );
            return BTreeMap::new();
        }
    };
    let raw: BTreeMap<String, serde_json::Value> = match serde_json::from_str(&contents) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "auth store file is not valid JSON, treating it as empty"
            );
            return BTreeMap::new();
        }
    };

    let mut credentials = BTreeMap::new();
    for (provider_key, value) in raw {
        let Some(provider) = OAuthProviderId::parse(&provider_key) else {
            tracing::warn!(provider = %provider_key, "unrecognized provider in auth store file, skipping");
            continue;
        };
        let disk: OnDiskCredential = match serde_json::from_value(value) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(provider = %provider_key, error = %e, "could not parse stored credential, skipping");
                continue;
            }
        };
        if disk.kind != "oauth" {
            tracing::warn!(
                provider = %provider_key,
                kind = %disk.kind,
                "unsupported credential type in auth store file, skipping"
            );
            continue;
        }
        credentials.insert(provider_key, on_disk_to_stored(provider, disk));
    }
    credentials
}

fn write_store_file(path: &Path, credentials: &BTreeMap<String, StoredCredential>) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    // `write_atomic` only *preserves* an existing file's permission bits — it doesn't impose one on a
    // brand-new file, so the very first write here would otherwise land at the umask default
    // (commonly world-readable). This file holds secrets; `settings.rs`/`trust_store.rs` never had
    // to care about this because they don't.
    if !path.exists() {
        create_private(path)?;
    }
    let mut raw = serde_json::Map::new();
    for (provider, stored) in credentials {
        let disk = credential_to_on_disk(&stored.credential, stored.last_refresh_error.clone());
        raw.insert(provider.clone(), serde_json::to_value(&disk).map_err(io::Error::other)?);
    }
    let body = serde_json::to_string_pretty(&raw).map_err(io::Error::other)?;
    let path_str = path
        .to_str()
        .ok_or_else(|| io::Error::other(format!("non-UTF-8 path: {}", path.display())))?;
    crate::tools::write_atomic(path_str, body.as_bytes())
}

/// Create an empty file at `path` with `0600` permissions set atomically at creation (Unix), not via
/// a `set_permissions` call afterward — closing the window where it would otherwise be
/// world/group-readable. Mirrors `session_store.rs`'s `create_private` helper (duplicated, not
/// shared, per this codebase's own small-self-contained-stores convention).
fn create_private(path: &Path) -> io::Result<()> {
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

/// How long [`FileLock::acquire`] retries before giving up.
const LOCK_TIMEOUT: Duration = Duration::from_secs(5);
/// How long between retries while waiting for another process's lock.
const LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(20);
/// A lock file older than this is assumed abandoned and is forcibly reclaimed instead of blocking
/// every future auth operation indefinitely.
const STALE_LOCK_AGE: Duration = Duration::from_secs(10);

/// A cross-process advisory lock via atomic lockfile creation — duplicated from
/// `trust_store.rs`/`settings.rs` rather than shared, matching this codebase's own established
/// convention that each store's lock is small and self-contained. Released by deleting the lock file
/// on `Drop`, so a panicked or early-returning holder still releases it.
struct FileLock {
    path: PathBuf,
}

impl FileLock {
    fn acquire(store_path: &Path) -> io::Result<Self> {
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
                        return Err(io::Error::new(
                            ErrorKind::TimedOut,
                            format!("timed out waiting for auth store lock at {}", lock_path.display()),
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
        .and_then(|modified| SystemTime::now().duration_since(modified).map_err(io::Error::other))
        .is_ok_and(|age| age > STALE_LOCK_AGE)
}

fn lock_path_for(store_path: &Path) -> PathBuf {
    let mut os = store_path.as_os_str().to_owned();
    os.push(".lock");
    PathBuf::from(os)
}

/// The default store path (`~/.claude/auth.json`, or `$AI_AGENT_CONFIG_DIR/auth.json`) — public so a
/// `CredentialSource` implementation (`crate::auth_credential_source`) built for a *different*
/// `AuthStore` handle than the one a caller already has open can still agree on the same file.
pub fn default_path() -> PathBuf {
    crate::settings::config_dir_root().join("auth.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anthropic_cred(access: &str, expires_at_ms: i64) -> OAuthCredential {
        OAuthCredential::Anthropic(AnthropicCredential {
            access: access.to_string(),
            refresh: "refresh-token".to_string(),
            expires_at_ms,
        })
    }

    #[test]
    fn missing_file_is_silently_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = AuthStore::open(dir.path().join("auth.json"));
        assert!(store.get("anthropic").is_none());
        assert_eq!(store.list().count(), 0);
    }

    #[test]
    fn corrupt_file_warns_and_is_treated_as_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        fs::write(&path, b"not json at all").unwrap();
        let store = AuthStore::open(path);
        assert!(store.get("anthropic").is_none());
    }

    #[test]
    fn set_then_get_round_trips_through_a_fresh_handle() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let mut store = AuthStore::open(path.clone());
        store.set("anthropic", anthropic_cred("access-1", 12345)).unwrap();

        let reopened = AuthStore::open(path);
        let stored = reopened.get("anthropic").expect("credential should be stored");
        assert_eq!(stored.credential.access(), "access-1");
        assert_eq!(stored.credential.expires_at_ms(), 12345);
        assert!(stored.last_refresh_error.is_none());
    }

    #[test]
    fn remove_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = AuthStore::open(dir.path().join("auth.json"));
        store.set("anthropic", anthropic_cred("a", 1)).unwrap();
        assert!(store.remove("anthropic").unwrap());
        assert!(!store.remove("anthropic").unwrap(), "second remove is a no-op, not an error");
        assert!(store.get("anthropic").is_none());
    }

    #[cfg(unix)]
    #[test]
    fn the_store_file_is_created_with_0600_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let mut store = AuthStore::open(path.clone());
        store.set("anthropic", anthropic_cred("a", 1)).unwrap();

        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "got mode {mode:o}");
    }

    #[test]
    fn refreshed_returns_not_logged_in_for_an_unknown_provider() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = AuthStore::open(dir.path().join("auth.json"));
        let result = store.refreshed("anthropic", |_| Ok(anthropic_cred("new", 999)));
        assert!(matches!(result, Err(AuthError::NotLoggedIn)));
    }

    #[test]
    fn refreshed_skips_the_network_call_when_not_actually_expired() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = AuthStore::open(dir.path().join("auth.json"));
        let far_future = now_ms() + 3_600_000;
        store.set("anthropic", anthropic_cred("still-valid", far_future)).unwrap();

        let calls = std::cell::Cell::new(0);
        let result = store.refreshed("anthropic", |_| {
            calls.set(calls.get() + 1);
            Ok(anthropic_cred("should-not-be-used", 0))
        });
        assert_eq!(calls.get(), 0, "refresh closure must not run for a non-expired credential");
        assert_eq!(result.unwrap().access(), "still-valid");
    }

    #[test]
    fn refreshed_calls_the_closure_and_persists_the_result_when_expired() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let mut store = AuthStore::open(path.clone());
        store.set("anthropic", anthropic_cred("old", now_ms() - 1000)).unwrap();

        let result = store
            .refreshed("anthropic", |_| Ok(anthropic_cred("refreshed", now_ms() + 3_600_000)))
            .unwrap();
        assert_eq!(result.access(), "refreshed");

        let reopened = AuthStore::open(path);
        assert_eq!(reopened.get("anthropic").unwrap().credential.access(), "refreshed");
    }

    #[test]
    fn refreshed_preserves_the_credential_and_records_the_error_on_failure() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let mut store = AuthStore::open(path.clone());
        store.set("anthropic", anthropic_cred("old", now_ms() - 1000)).unwrap();

        let result = store.refreshed("anthropic", |_| Err("network unreachable".to_string()));
        assert!(matches!(result, Err(AuthError::RefreshFailed(_))));

        let reopened = AuthStore::open(path);
        let stored = reopened.get("anthropic").unwrap();
        // The credential itself must be untouched — a transient failure must not lose it.
        assert_eq!(stored.credential.access(), "old");
        assert_eq!(
            stored.last_refresh_error.as_deref(),
            Some("network unreachable")
        );
    }

    #[test]
    fn a_stale_in_memory_snapshot_is_refreshed_before_mutating_not_blindly_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("auth.json");

        let mut handle_a = AuthStore::open(store_path.clone());
        let mut handle_b = AuthStore::open(store_path.clone());

        handle_a.set("anthropic", anthropic_cred("a", 1)).unwrap();
        handle_b.set("github-copilot", anthropic_cred("b", 2)).unwrap();

        let reopened = AuthStore::open(store_path);
        assert!(
            reopened.get("anthropic").is_some(),
            "provider a's credential must survive provider b's later write from a stale handle"
        );
        assert!(reopened.get("github-copilot").is_some());
    }

    #[test]
    fn concurrent_set_calls_from_separate_handles_do_not_lose_an_update() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("auth.json");
        const PROVIDERS: [&str; 3] = ["anthropic", "github-copilot", "openai-codex"];
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(PROVIDERS.len()));

        let handles: Vec<_> = PROVIDERS
            .iter()
            .map(|provider| {
                let provider = provider.to_string();
                let store_path = store_path.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let mut store = AuthStore::open(store_path);
                    barrier.wait();
                    store.set(&provider, anthropic_cred("tok", 1)).unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        let reopened = AuthStore::open(store_path);
        for provider in PROVIDERS {
            assert!(
                reopened.get(provider).is_some(),
                "{provider} lost its credential to a concurrent write"
            );
        }
    }

    #[test]
    fn a_stale_lock_file_is_reclaimed_rather_than_blocking_forever() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("auth.json");
        let lock_path = lock_path_for(&store_path);
        fs::write(&lock_path, b"").unwrap();
        let stale = SystemTime::now() - (STALE_LOCK_AGE + Duration::from_secs(1));
        fs::File::open(&lock_path).unwrap().set_modified(stale).unwrap();

        let mut store = AuthStore::open(store_path);
        store.set("anthropic", anthropic_cred("a", 1)).unwrap();
        assert!(store.get("anthropic").is_some());
    }
}
