//! [`OAuthCredentialSource`] — the `agent_core::client::CredentialSource` implementation that makes
//! an OAuth login usable as the credential for a `GatewayClient` request, transparently refreshing
//! an expiring token via `AuthStore`. This is the one piece of `agent_core`'s `CredentialSource` seam
//! this crate actually implements — see that trait's doc comment for why the trait itself lives in
//! `agent_core` but this implementation must live here instead.

use std::path::PathBuf;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::auth_store::AuthStore;
use crate::oauth::{self, OAuthProviderId};

/// Resolves a live, non-expired access token immediately before each request.
pub struct OAuthCredentialSource {
    provider: OAuthProviderId,
    store_path: PathBuf,
    /// In-memory fast path: avoids a disk read + cross-process file lock on the overwhelmingly
    /// common case where the last-fetched token is still valid.
    cached: Mutex<Option<(String, i64)>>,
}

impl OAuthCredentialSource {
    pub fn new(provider: OAuthProviderId, store_path: PathBuf) -> Self {
        Self {
            provider,
            store_path,
            cached: Mutex::new(None),
        }
    }
}

#[async_trait]
impl agent_core::client::CredentialSource for OAuthCredentialSource {
    async fn credential(&self) -> agent_core::Result<agent_core::client::Credential> {
        let now = now_ms();
        if let Some((access, expires_at_ms)) = cached_snapshot(&self.cached) {
            if now < expires_at_ms {
                return Ok(agent_core::client::Credential::new(access, true));
            }
        }

        let provider = self.provider;
        let store_path = self.store_path.clone();
        // `AuthStore::refreshed` holds a cross-process file lock across the *entire* refresh
        // (including the network call — see that method's doc comment for why), which can block for
        // up to several seconds under contention. That must not run directly on a tokio worker
        // thread — `spawn_blocking` is the same fix this codebase already applies to
        // `serve.rs`'s `persist_blocking` for an analogous blocking-write concern.
        let refreshed = tokio::task::spawn_blocking(move || {
            let mut store = AuthStore::open(store_path);
            store.refreshed(provider.store_key(), |cred| {
                // Bridges the one async sub-step (the actual token-endpoint call) back into this
                // synchronous closure — sanctioned precisely because we're already on a
                // `spawn_blocking` thread, not a tokio async worker thread.
                tokio::runtime::Handle::current()
                    .block_on(oauth::refresh(cred))
                    .map_err(|e| e.to_string())
            })
        })
        .await
        .map_err(|e| agent_core::Error::Transport(format!("credential refresh task panicked: {e}")))?;

        let credential = refreshed.map_err(|e| {
            agent_core::Error::Transport(format!(
                "OAuth credential for \"{provider}\" unavailable: {e}. Run `agent login {provider}` \
                 to re-authenticate."
            ))
        })?;

        let access = credential.access().to_string();
        let expires_at_ms = credential.expires_at_ms();
        *self.cached.lock().unwrap_or_else(|e| e.into_inner()) = Some((access.clone(), expires_at_ms));
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
        let source = OAuthCredentialSource::new(OAuthProviderId::Anthropic, dir.path().join("auth.json"));
        let err = match source.credential().await {
            Ok(_) => panic!("expected an error when nothing is stored"),
            Err(e) => e,
        };
        match err {
            agent_core::Error::Transport(msg) => {
                assert!(msg.contains("agent login anthropic"), "got: {msg}")
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
}
