//! [`OAuthCredentialSource`] — the `agent_core::client::CredentialSource` implementation that makes
//! an OAuth login usable as the credential for a `GatewayClient` request, transparently refreshing
//! an expiring token via `AuthStore`. This is the one piece of `agent_core`'s `CredentialSource` seam
//! this crate actually implements — see that trait's doc comment for why the trait itself lives in
//! `agent_core` but this implementation must live here instead.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;

use crate::auth_store::AuthStore;
use crate::oauth::{self, OAuthCredential, OAuthProviderId};
use crate::tools::mcp_oauth::{FIRST_BACKOFF, MAX_BACKOFF};

/// The token-endpoint call, as the (synchronous) closure `AuthStore::refreshed` takes. A seam so a
/// test can count and fail it without a network.
type RefreshFn = Arc<dyn Fn(&OAuthCredential) -> Result<OAuthCredential, String> + Send + Sync>;

/// A refresh that failed, remembered so the next request does not repeat the file-locked network
/// call while the authorization server is down: the same doubling backoff an MCP server's OAuth
/// refresh uses (`mcp_oauth`: 30 s, doubling to 5 min).
struct RefreshFailure {
    message: String,
    until: Instant,
    backoff: Duration,
}

/// Resolves a live, non-expired access token immediately before each request.
pub struct OAuthCredentialSource {
    provider: OAuthProviderId,
    store_path: PathBuf,
    /// In-memory fast path: avoids a disk read + cross-process file lock on the overwhelmingly
    /// common case where the last-fetched token is still valid.
    cached: Mutex<Option<(String, i64)>>,
    /// The last refresh failure, while its backoff stands. Cleared by a successful refresh, and
    /// bypassed when the store holds a live token again (another process refreshed, or `login`).
    failed: Mutex<Option<RefreshFailure>>,
    first_backoff: Duration,
    refresh: RefreshFn,
}

impl OAuthCredentialSource {
    pub fn new(provider: OAuthProviderId, store_path: PathBuf) -> Self {
        Self {
            provider,
            store_path,
            cached: Mutex::new(None),
            failed: Mutex::new(None),
            first_backoff: FIRST_BACKOFF,
            refresh: Arc::new(|cred| {
                // Bridges the one async sub-step (the actual token-endpoint call) back into this
                // synchronous closure — sanctioned precisely because `credential` runs it on a
                // `spawn_blocking` thread, not a tokio async worker thread.
                tokio::runtime::Handle::current()
                    .block_on(oauth::refresh(cred))
                    .map_err(|e| e.to_string())
            }),
        }
    }

    /// The remembered failure's error, if its backoff still stands and the store has nothing newer
    /// to offer: a token that is live again (another process refreshed it, or the user logged in
    /// again) ends the backoff at once. A failure past its backoff is retried.
    fn standing_failure(&self, now: Instant) -> Option<agent_core::Error> {
        let message = {
            let failed = self.failed.lock().unwrap_or_else(|e| e.into_inner());
            let f = failed.as_ref().filter(|f| now < f.until)?;
            f.message.clone()
        };
        let stored = AuthStore::open(self.store_path.clone());
        let live = stored
            .get(self.provider.store_key())
            .is_none_or(|s| !s.credential.is_expired(now_ms()));
        (!live).then_some(agent_core::Error::Transport(message))
    }
}

#[async_trait]
impl agent_core::client::CredentialSource for OAuthCredentialSource {
    async fn credential(&self) -> agent_core::Result<agent_core::client::Credential> {
        let now = now_ms();
        if let Some((access, expires_at_ms)) = cached_snapshot(&self.cached)
            && now < expires_at_ms
        {
            return Ok(agent_core::client::Credential::new(access, true));
        }

        if let Some(e) = self.standing_failure(Instant::now()) {
            return Err(e);
        }

        let provider = self.provider;
        let store_path = self.store_path.clone();
        let refresh = self.refresh.clone();
        // `AuthStore::refreshed` holds a cross-process file lock across the *entire* refresh
        // (including the network call — see that method's doc comment for why), which can block for
        // up to several seconds under contention. That must not run directly on a tokio worker
        // thread — `spawn_blocking` is the same fix this codebase already applies to
        // `serve/persistence.rs`'s `persist_blocking` for an analogous blocking-write concern.
        let refreshed = tokio::task::spawn_blocking(move || {
            let mut store = AuthStore::open(store_path);
            store.refreshed(provider.store_key(), |cred| refresh(cred))
        })
        .await
        .map_err(|e| {
            agent_core::Error::Transport(format!("credential refresh task panicked: {e}"))
        })?;

        let credential = match refreshed {
            Ok(credential) => {
                *self.failed.lock().unwrap_or_else(|e| e.into_inner()) = None;
                credential
            }
            Err(e) => {
                let message = format!(
                    "OAuth credential for \"{provider}\" unavailable: {e}. Run `{} login {provider}` \
                     to re-authenticate.",
                    crate::PROGRAM
                );
                // Only a failed refresh (the network call) is remembered; "not logged in" and a
                // store I/O error cost nothing to find out again.
                if matches!(e, crate::auth_store::AuthError::RefreshFailed(_)) {
                    let mut failed = self.failed.lock().unwrap_or_else(|e| e.into_inner());
                    let backoff = match failed.as_ref() {
                        Some(f) => (f.backoff * 2).min(MAX_BACKOFF),
                        None => self.first_backoff,
                    };
                    *failed = Some(RefreshFailure {
                        message: message.clone(),
                        until: Instant::now() + backoff,
                        backoff,
                    });
                }
                return Err(agent_core::Error::Transport(message));
            }
        };

        let access = credential.access().to_string();
        let expires_at_ms = credential.expires_at_ms();
        *self.cached.lock().unwrap_or_else(|e| e.into_inner()) =
            Some((access.clone(), expires_at_ms));
        Ok(agent_core::client::Credential::new(access, true))
    }
}

fn cached_snapshot(cached: &Mutex<Option<(String, i64)>>) -> Option<(String, i64)> {
    cached.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::client::CredentialSource;

    #[tokio::test]
    async fn returns_not_logged_in_as_a_transport_error_when_nothing_is_stored() {
        let dir = tempfile::tempdir().unwrap();
        let source =
            OAuthCredentialSource::new(OAuthProviderId::Anthropic, dir.path().join("auth.json"));
        let err = match source.credential().await {
            Ok(_) => panic!("expected an error when nothing is stored"),
            Err(e) => e,
        };
        match err {
            agent_core::Error::Transport(msg) => {
                assert!(
                    msg.contains(&format!("`{} login anthropic`", crate::PROGRAM)),
                    "got: {msg}"
                )
            }
            other => panic!("expected Transport error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_cached_non_expired_token_never_touches_the_store() {
        let dir = tempfile::tempdir().unwrap();
        // A directory, not a file — any attempt to `AuthStore::open`/read this path would misbehave;
        // proves the cached fast path genuinely skips storage entirely.
        let bogus_path = dir.path().join("not-a-real-file").join("auth.json");
        let source = OAuthCredentialSource::new(OAuthProviderId::Anthropic, bogus_path);
        *source.cached.lock().unwrap() = Some(("cached-token".to_string(), now_ms() + 3_600_000));

        let credential = source.credential().await.unwrap();
        // `Credential`'s fields are private; the only externally observable proof available here is
        // that no error occurred despite a store path that would fail to open for a real read.
        drop(credential);
    }

    fn expired_store(dir: &tempfile::TempDir) -> PathBuf {
        use crate::oauth::anthropic::AnthropicCredential;
        let path = dir.path().join("auth.json");
        AuthStore::open(path.clone())
            .set(
                "anthropic",
                OAuthCredential::Anthropic(AnthropicCredential {
                    access: "old".into(),
                    refresh: "r".into(),
                    expires_at_ms: now_ms() - 1_000,
                }),
            )
            .unwrap();
        path
    }

    fn failing_source(
        path: PathBuf,
        first_backoff: Duration,
    ) -> (OAuthCredentialSource, Arc<std::sync::atomic::AtomicUsize>) {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = calls.clone();
        let mut source = OAuthCredentialSource::new(OAuthProviderId::Anthropic, path);
        source.first_backoff = first_backoff;
        source.refresh = Arc::new(move |_| {
            counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err("connection refused".into())
        });
        (source, calls)
    }

    #[tokio::test]
    async fn a_failed_refresh_is_not_repeated_on_every_request() {
        let dir = tempfile::tempdir().unwrap();
        let (source, calls) = failing_source(expired_store(&dir), Duration::from_secs(30));
        for _ in 0..3 {
            assert!(source.credential().await.is_err());
        }
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the authorization server is asked once, then the failure stands for its backoff"
        );
    }

    #[tokio::test]
    async fn a_failed_refresh_is_retried_after_its_backoff_which_then_doubles() {
        let dir = tempfile::tempdir().unwrap();
        let (source, calls) = failing_source(expired_store(&dir), Duration::from_millis(100));
        let n = || calls.load(std::sync::atomic::Ordering::SeqCst);
        assert!(source.credential().await.is_err());
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(source.credential().await.is_err());
        assert_eq!(n(), 2, "past the first backoff it is retried");
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(source.credential().await.is_err());
        assert_eq!(
            n(),
            2,
            "the second backoff is doubled: 150 ms is inside 200 ms"
        );
    }

    #[tokio::test]
    async fn a_live_token_in_the_store_ends_the_backoff() {
        use crate::oauth::anthropic::AnthropicCredential;
        let dir = tempfile::tempdir().unwrap();
        let path = expired_store(&dir);
        let (source, calls) = failing_source(path.clone(), Duration::from_secs(30));
        assert!(source.credential().await.is_err());
        // Another process refreshed, or the user logged in again.
        AuthStore::open(path)
            .set(
                "anthropic",
                OAuthCredential::Anthropic(AnthropicCredential {
                    access: "new".into(),
                    refresh: "r2".into(),
                    expires_at_ms: now_ms() + 3_600_000,
                }),
            )
            .unwrap();
        source.credential().await.expect("the live token is used");
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
