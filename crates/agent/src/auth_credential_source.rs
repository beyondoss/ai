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

/// The identity of the store file at one moment, from a single `stat`. `AuthStore` writes by
/// temp-file-and-rename, so every write gives the path a new inode; size, mtime and ctime (both to
/// the nanosecond) also catch a write made in place. A missing file is `None`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct StoreStamp {
    #[cfg(unix)]
    ids: (u64, u64, i64, i64),
    len: u64,
    modified: Option<std::time::SystemTime>,
}

impl StoreStamp {
    fn of(path: &std::path::Path) -> Option<Self> {
        let meta = std::fs::metadata(path).ok()?;
        #[cfg(unix)]
        let ids = {
            use std::os::unix::fs::MetadataExt;
            (meta.dev(), meta.ino(), meta.ctime(), meta.ctime_nsec())
        };
        Some(Self {
            #[cfg(unix)]
            ids,
            len: meta.len(),
            modified: meta.modified().ok(),
        })
    }
}

/// The token the last credential fetch returned, and the store file it was read from.
struct Cached {
    access: String,
    expires_at_ms: i64,
    /// Taken before the store was read, so a write racing the read makes the cache look stale, never
    /// fresh.
    stamp: StoreStamp,
}

/// Resolves a live, non-expired access token immediately before each request.
///
/// The on-disk store is the source of truth; `cached` is a copy of it. The copy is used only while
/// the store file is unchanged since it was filled (one `stat`, no read) and the provider has not
/// rejected the token ([`rejected`](agent_core::client::CredentialSource::rejected)).
pub struct OAuthCredentialSource {
    provider: OAuthProviderId,
    store_path: PathBuf,
    /// In-memory fast path: avoids a disk read + cross-process file lock on the overwhelmingly
    /// common case where the store is unchanged and its token still valid.
    cached: Mutex<Option<Cached>>,
    /// A token the provider answered with 401. While the store still holds it, the next fetch
    /// refreshes it even though its expiry says it is live.
    rejected: Mutex<Option<String>>,
    /// How many times a fetch left the fast path (test-only: proves the fast path reads nothing).
    #[cfg(test)]
    slow_paths: std::sync::atomic::AtomicUsize,
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
            rejected: Mutex::new(None),
            #[cfg(test)]
            slow_paths: std::sync::atomic::AtomicUsize::new(0),
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
        let rejected = self
            .rejected
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let live = stored.get(self.provider.store_key()).is_none_or(|s| {
            !s.credential.is_expired(now_ms()) && rejected.as_deref() != Some(s.credential.access())
        });
        (!live).then_some(agent_core::Error::Transport(message))
    }
}

#[async_trait]
impl agent_core::client::CredentialSource for OAuthCredentialSource {
    async fn credential(&self) -> agent_core::Result<agent_core::client::Credential> {
        // One `stat`, before anything is read: the stamp a refill below stores is never newer than
        // what it read.
        let stamp = StoreStamp::of(&self.store_path);
        {
            let mut cached = self.cached.lock().unwrap_or_else(|e| e.into_inner());
            match cached.as_ref() {
                Some(c) if Some(c.stamp) == stamp && now_ms() < c.expires_at_ms => {
                    return Ok(agent_core::client::Credential::new(c.access.clone(), true));
                }
                // Changed (login or logout elsewhere, a refresh by another process), gone, or
                // expired: the copy is not used again.
                _ => *cached = None,
            }
        }
        #[cfg(test)]
        self.slow_paths
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        if let Some(e) = self.standing_failure(Instant::now()) {
            return Err(e);
        }

        let provider = self.provider;
        let store_path = self.store_path.clone();
        let refresh = self.refresh.clone();
        let rejected = self
            .rejected
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        // `AuthStore::refreshed` holds a cross-process file lock across the *entire* refresh
        // (including the network call — see that method's doc comment for why), which can block for
        // up to several seconds under contention. That must not run directly on a tokio worker
        // thread — `spawn_blocking` is the same fix this codebase already applies to
        // `serve/persistence.rs`'s `persist_blocking` for an analogous blocking-write concern.
        let refreshed = tokio::task::spawn_blocking(move || {
            let mut store = AuthStore::open(store_path);
            store.refreshed_rejecting(provider.store_key(), rejected.as_deref(), |cred| {
                refresh(cred)
            })
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
        // The token the store handed back is not the rejected one (or the refresh would have
        // replaced it); the rejection is spent.
        {
            let mut rejected = self.rejected.lock().unwrap_or_else(|e| e.into_inner());
            if rejected.as_deref() != Some(access.as_str()) {
                *rejected = None;
            }
        }
        // No stamp (the file vanished again) means nothing is cached.
        if let Some(stamp) = stamp {
            *self.cached.lock().unwrap_or_else(|e| e.into_inner()) = Some(Cached {
                access: access.clone(),
                expires_at_ms,
                stamp,
            });
        }
        Ok(agent_core::client::Credential::new(access, true))
    }

    fn rejected(&self, token: &str) {
        let mut cached = self.cached.lock().unwrap_or_else(|e| e.into_inner());
        // A different cached token is a newer one (another request already refreshed): keep it.
        if cached.as_ref().is_some_and(|c| c.access != token) {
            return;
        }
        *cached = None;
        *self.rejected.lock().unwrap_or_else(|e| e.into_inner()) = Some(token.to_string());
    }
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

    use crate::oauth::anthropic::AnthropicCredential;
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

    fn anthropic(access: &str, refresh: &str) -> OAuthCredential {
        OAuthCredential::Anthropic(AnthropicCredential {
            access: access.into(),
            refresh: refresh.into(),
            expires_at_ms: now_ms() + 3_600_000,
        })
    }

    /// A source over a store holding a live `old` token, whose refresh call hands out `new` and
    /// counts itself.
    fn live_source(dir: &tempfile::TempDir) -> (OAuthCredentialSource, PathBuf, Arc<AtomicUsize>) {
        let path = dir.path().join("auth.json");
        AuthStore::open(path.clone())
            .set("anthropic", anthropic("old", "r"))
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let mut source = OAuthCredentialSource::new(OAuthProviderId::Anthropic, path.clone());
        source.refresh = Arc::new(move |_| {
            counted.fetch_add(1, SeqCst);
            Ok(anthropic("new", "r2"))
        });
        (source, path, calls)
    }

    async fn token(source: &OAuthCredentialSource) -> agent_core::Result<String> {
        // `Credential`'s key is private to agent_core; a successful fetch leaves the token it
        // returned in the cache.
        source.credential().await?;
        let cached = source.cached.lock().unwrap();
        Ok(cached
            .as_ref()
            .expect("a fetch fills the cache")
            .access
            .clone())
    }

    #[tokio::test]
    async fn an_unchanged_store_is_served_from_the_cache_without_a_read_or_refresh() {
        let dir = tempfile::tempdir().unwrap();
        let (source, _path, calls) = live_source(&dir);
        assert_eq!(token(&source).await.unwrap(), "old");
        let after_fill = source.slow_paths.load(SeqCst);
        for _ in 0..5 {
            assert_eq!(token(&source).await.unwrap(), "old");
        }
        assert_eq!(source.slow_paths.load(SeqCst), after_fill, "no slow path");
        assert_eq!(calls.load(SeqCst), 0, "no refresh");
    }

    #[tokio::test]
    async fn a_logout_in_another_process_is_not_served_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let (source, path, calls) = live_source(&dir);
        assert_eq!(token(&source).await.unwrap(), "old");
        AuthStore::open(path).remove("anthropic").unwrap();
        let err = token(&source).await.expect_err("logged out");
        assert!(format!("{err}").contains("login anthropic"), "got: {err}");
        assert_eq!(calls.load(SeqCst), 0, "no network call");
    }

    #[tokio::test]
    async fn a_deleted_store_file_is_not_served_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let (source, path, calls) = live_source(&dir);
        assert_eq!(token(&source).await.unwrap(), "old");
        std::fs::remove_file(path).unwrap();
        assert!(token(&source).await.is_err());
        assert_eq!(calls.load(SeqCst), 0);
    }

    #[tokio::test]
    async fn a_login_in_another_process_replaces_the_cached_token() {
        let dir = tempfile::tempdir().unwrap();
        let (source, path, calls) = live_source(&dir);
        assert_eq!(token(&source).await.unwrap(), "old");
        AuthStore::open(path)
            .set("anthropic", anthropic("relogged", "r3"))
            .unwrap();
        assert_eq!(token(&source).await.unwrap(), "relogged");
        assert_eq!(calls.load(SeqCst), 0);
    }

    #[tokio::test]
    async fn a_rejected_token_is_refreshed_even_though_its_expiry_is_live() {
        let dir = tempfile::tempdir().unwrap();
        let (source, _path, calls) = live_source(&dir);
        assert_eq!(token(&source).await.unwrap(), "old");
        source.rejected("old");
        assert_eq!(token(&source).await.unwrap(), "new");
        assert_eq!(calls.load(SeqCst), 1);
        // The refreshed token is cached again, and not refreshed twice.
        assert_eq!(token(&source).await.unwrap(), "new");
        assert_eq!(token(&source).await.unwrap(), "new");
        assert_eq!(calls.load(SeqCst), 1);
    }

    #[tokio::test]
    async fn a_rejection_of_a_token_already_replaced_changes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let (source, _path, calls) = live_source(&dir);
        assert_eq!(token(&source).await.unwrap(), "old");
        source.rejected("some-older-token");
        assert_eq!(token(&source).await.unwrap(), "old");
        assert_eq!(calls.load(SeqCst), 0);
    }

    #[tokio::test]
    async fn another_process_having_refreshed_the_rejected_token_is_used_not_refreshed_again() {
        let dir = tempfile::tempdir().unwrap();
        let (source, path, calls) = live_source(&dir);
        assert_eq!(token(&source).await.unwrap(), "old");
        AuthStore::open(path)
            .set("anthropic", anthropic("theirs", "r3"))
            .unwrap();
        source.rejected("old");
        assert_eq!(token(&source).await.unwrap(), "theirs");
        assert_eq!(calls.load(SeqCst), 0);
    }

    /// A provider that answers 401 to the token in `reject` and 200 to any other, recording the
    /// bearer of every request it gets.
    struct FakeProvider {
        addr: std::net::SocketAddr,
        seen: Arc<Mutex<Vec<String>>>,
        stop: Arc<std::sync::atomic::AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl FakeProvider {
        fn start(reject: &'static [&'static str]) -> Self {
            use std::io::{Read, Write};
            let listener = beyond_test_support::ports::listener();
            listener.set_nonblocking(true).unwrap();
            let addr = listener.local_addr().unwrap();
            let seen = Arc::new(Mutex::new(Vec::new()));
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let (seen2, stop2) = (seen.clone(), stop.clone());
            let thread = std::thread::spawn(move || {
                while !stop2.load(SeqCst) {
                    let Ok((mut conn, _)) = listener.accept() else {
                        std::thread::sleep(Duration::from_millis(5));
                        continue;
                    };
                    conn.set_nonblocking(false).unwrap();
                    conn.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    let end = loop {
                        let n = conn.read(&mut chunk).unwrap();
                        buf.extend_from_slice(&chunk[..n]);
                        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            break i + 4;
                        }
                        assert!(n > 0, "connection closed mid-headers");
                    };
                    let head = String::from_utf8_lossy(&buf[..end]).to_lowercase();
                    let len = head
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length: "))
                        .and_then(|v| v.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    while buf.len() < end + len {
                        let n = conn.read(&mut chunk).unwrap();
                        assert!(n > 0, "connection closed mid-body");
                        buf.extend_from_slice(&chunk[..n]);
                    }
                    let bearer = head
                        .lines()
                        .find_map(|l| l.strip_prefix("authorization: bearer "))
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    let response: &[u8] = if reject.contains(&bearer.as_str()) {
                        b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}"
                    } else {
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
                    };
                    seen2.lock().unwrap().push(bearer);
                    let _ = conn.write_all(response);
                }
            });
            Self {
                addr,
                seen,
                stop,
                thread: Some(thread),
            }
        }

        fn seen(&self) -> Vec<String> {
            self.seen.lock().unwrap().clone()
        }
    }

    impl Drop for FakeProvider {
        fn drop(&mut self) {
            self.stop.store(true, SeqCst);
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
        }
    }

    async fn drain(
        source: Arc<OAuthCredentialSource>,
        provider: &FakeProvider,
    ) -> agent_core::Result<()> {
        use agent_core::ModelTransport as _;
        use futures::StreamExt;
        let client = agent_core::client::GatewayClient::with_credential_source(
            format!("http://{}", provider.addr),
            source,
        )
        .unwrap();
        let req = agent_core::ModelRequest::new("claude-test", Vec::new(), 100);
        let mut events = client.stream(req).await?;
        while let Some(ev) = events.next().await {
            ev?;
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_401_refreshes_the_token_and_the_request_is_retried_once_with_the_new_one() {
        let dir = tempfile::tempdir().unwrap();
        let (source, _path, calls) = live_source(&dir);
        let provider = FakeProvider::start(&["old"]);
        drain(Arc::new(source), &provider)
            .await
            .expect("retry succeeds");
        assert_eq!(provider.seen(), ["old", "new"]);
        assert_eq!(calls.load(SeqCst), 1);
    }

    #[tokio::test]
    async fn two_401s_are_one_retry_then_the_error() {
        let dir = tempfile::tempdir().unwrap();
        let (source, _path, calls) = live_source(&dir);
        let provider = FakeProvider::start(&["old", "new"]);
        let err = drain(Arc::new(source), &provider).await.expect_err("401");
        assert!(format!("{err}").contains("401"), "got: {err}");
        assert_eq!(provider.seen(), ["old", "new"], "exactly one retry");
        assert_eq!(calls.load(SeqCst), 1);
    }
}
