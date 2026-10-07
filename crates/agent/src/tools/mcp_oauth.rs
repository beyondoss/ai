//! Mid-session MCP OAuth: a server that answers **401** to a request (its token expired, or was
//! revoked, while the session was running) gets one token refresh and one retry, instead of failing
//! every request until the server is redialed.
//!
//! - [`ServerAuth`] is one configured server's bearer token, shared by **every** connection to it in
//!   this process (the plain and the apps-flavoured connections, and direct `events/*` requests,
//!   which read it per request). It is reloaded from the `agent mcp-login` store at each dial, under
//!   the same lock a refresh holds, so a reload can never put back a token a refresh just replaced.
//! - On a 401, [`ServerAuth::after_rejection`] refreshes through the same
//!   [`AuthorizationManager`](rmcp::transport::auth::AuthorizationManager) and
//!   [`McpAuthStore`](crate::mcp_auth_store::McpAuthStore) `mcp-login` uses — so the refreshed token
//!   is persisted, and the next process starts from it. **Single flight:** the refresh runs under the
//!   token's lock, and a caller whose rejected token is no longer current takes the new one without
//!   refreshing again, so any number of concurrent 401s cost one refresh (which also matters for an
//!   authorization server that rotates refresh tokens: a second refresh with the spent one would
//!   fail).
//! - **One rule for what triggers it:** any 401 from a server with a login — with or without a
//!   `WWW-Authenticate` challenge, on rmcp's path ([`OAuthHttp`]) and on the direct `events/*` path
//!   alike. A 403 (`InsufficientScope`) never does: a refresh does not widen scopes.
//! - [`OAuthHttp`] applies that to every request rmcp's streamable-HTTP transport makes —
//!   `tools/call`, `resources/*`, `prompts/*`, `skills/*`, MCP App view reads, the handshake, the
//!   standalone SSE stream — retrying each **once**.
//! - **A failed refresh** is remembered so the authorization server is not asked again on every
//!   request: a *definitive* one (the refresh token was rejected — `invalid_grant` — or there is
//!   none) for good, with an error naming `agent mcp-login <server>`; a *transient* one (the
//!   authorization server unreachable, a 5xx) only for a backoff — 30 s, doubling to 5 min — after
//!   which a rejection refreshes again.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::stream::BoxStream;
use http::{HeaderName, HeaderValue};
use rmcp::model::ClientJsonRpcMessage;
use rmcp::transport::streamable_http_client::{
    SseError, StreamableHttpClient, StreamableHttpError, StreamableHttpPostResponse,
};

/// The first wait after a transient refresh failure; each further one doubles it, up to
/// [`MAX_BACKOFF`].
const FIRST_BACKOFF: Duration = Duration::from_secs(30);
const MAX_BACKOFF: Duration = Duration::from_secs(300);

/// Why a refresh did not produce a token.
#[derive(Debug, Clone)]
pub(crate) struct RefreshError {
    /// The authorization server rejected the refresh token, or there is none: only a new
    /// `agent mcp-login` helps. Otherwise transient (it may work later).
    pub(crate) definitive: bool,
    pub(crate) reason: String,
}

impl RefreshError {
    pub(crate) fn definitive(reason: impl Into<String>) -> Self {
        Self {
            definitive: true,
            reason: reason.into(),
        }
    }

    pub(crate) fn transient(reason: impl Into<String>) -> Self {
        Self {
            definitive: false,
            reason: reason.into(),
        }
    }

    fn from_auth(e: rmcp::transport::auth::AuthError) -> Self {
        use rmcp::transport::auth::AuthError;
        match e {
            AuthError::TokenRefreshRejected(_) | AuthError::AuthorizationRequired => {
                Self::definitive(e.to_string())
            }
            other => Self::transient(other.to_string()),
        }
    }
}

#[cfg(test)]
type FakeRefresh = Arc<dyn Fn() -> Result<String, RefreshError> + Send + Sync>;
#[cfg(test)]
type FakeStored = Arc<dyn Fn() -> Option<String> + Send + Sync>;

/// One configured server's OAuth bearer token, shared by every connection to it. See the module doc.
pub(crate) struct ServerAuth {
    server: String,
    url: String,
    state: tokio::sync::Mutex<TokenState>,
    first_backoff: Duration,
    /// Stand-ins for the authorization server and the store, for unit tests (`None`: the real ones).
    #[cfg(test)]
    fake_refresh: Option<FakeRefresh>,
    #[cfg(test)]
    fake_stored: Option<FakeStored>,
}

#[derive(Default)]
struct TokenState {
    /// What every request to the server sends; `None` when the stored login yields no usable token.
    token: Option<String>,
    /// The last refresh after `token` was rejected failed.
    failed: Option<Failed>,
}

struct Failed {
    /// What a rejection of the same token returns while this stands.
    message: String,
    /// `None`: for good (a definitive failure). Otherwise when a refresh may be tried again.
    until: Option<Instant>,
    /// The backoff that produced `until`; the next transient failure doubles it.
    backoff: Duration,
}

/// Every server's [`ServerAuth`] in this process, by `(name, url)`.
static SERVERS: Mutex<Vec<Arc<ServerAuth>>> = Mutex::new(Vec::new());

impl ServerAuth {
    fn new(server: &str, url: &str) -> Self {
        Self {
            server: server.to_owned(),
            url: url.to_owned(),
            state: tokio::sync::Mutex::new(TokenState::default()),
            first_backoff: FIRST_BACKOFF,
            #[cfg(test)]
            fake_refresh: None,
            #[cfg(test)]
            fake_stored: None,
        }
    }

    /// The shared auth for configured server `name` at `url`, with its token (re)loaded from the
    /// `agent mcp-login` store — refreshed first if it is about to expire — or `None` when nobody
    /// has logged in to it (the common case: a single file check, no network).
    pub(crate) async fn load(name: &str, url: &str) -> Option<Arc<Self>> {
        if !crate::mcp_auth_store::McpAuthStore::open_default().has_credential(name) {
            return None;
        }
        let auth = {
            let mut servers = SERVERS.lock().unwrap_or_else(|e| e.into_inner());
            match servers.iter().find(|a| a.server == name && a.url == url) {
                Some(auth) => auth.clone(),
                None => {
                    let auth = Arc::new(Self::new(name, url));
                    servers.push(auth.clone());
                    auth
                }
            }
        };
        auth.reload().await;
        Some(auth)
    }

    /// Reload the token from the store, **under the token's lock**: a refresh persists its token
    /// before it releases the lock, so what the store holds here is at least as new as what this
    /// process holds — never an older token read before a refresh finished and written over it
    /// after. When the store yields nothing (unreadable, or its refresh failed) the token held is
    /// kept. A login (or logout) made since the last dial takes effect here, as it always did.
    async fn reload(&self) {
        let mut state = self.state.lock().await;
        let Some(stored) = self.stored().await else {
            return;
        };
        if state.token.as_deref() != Some(stored.as_str()) {
            state.failed = None;
            state.token = Some(stored);
        }
    }

    async fn stored(&self) -> Option<String> {
        #[cfg(test)]
        if let Some(fake) = &self.fake_stored {
            return fake();
        }
        stored_token(&self.server, &self.url).await
    }

    async fn refresh(&self) -> Result<String, RefreshError> {
        #[cfg(test)]
        if let Some(fake) = &self.fake_refresh {
            return fake();
        }
        refresh(&self.server, &self.url).await
    }

    /// The token requests send now.
    pub(crate) async fn token(&self) -> Option<String> {
        self.state.lock().await.token.clone()
    }

    /// `sent` was answered 401: the token to retry with. If another caller already replaced `sent`,
    /// that replacement — no second refresh. Otherwise one refresh, persisted. Its failure is the
    /// `Err` and is remembered (see the module doc): for good when definitive, naming
    /// `agent mcp-login`; for a doubling backoff when transient.
    pub(crate) async fn after_rejection(&self, sent: Option<&str>) -> Result<String, String> {
        let mut state = self.state.lock().await;
        if let Some(current) = state.token.as_deref()
            && Some(current) != sent
        {
            return Ok(current.to_owned());
        }
        let now = Instant::now();
        if let Some(failed) = &state.failed
            && failed.until.is_none_or(|until| now < until)
        {
            return Err(failed.message.clone());
        }
        match self.refresh().await {
            Ok(token) => {
                tracing::info!(server = %self.server, "refreshed a rejected MCP OAuth token");
                state.token = Some(token.clone());
                state.failed = None;
                Ok(token)
            }
            Err(e) if e.definitive => {
                let message = format!(
                    "mcp server `{0}` rejected its OAuth token and it could not be refreshed ({1}); \
                     run `agent mcp-login {0}` again",
                    self.server, e.reason
                );
                tracing::warn!(server = %self.server, error = %e.reason, "MCP OAuth refresh rejected");
                state.failed = Some(Failed {
                    message: message.clone(),
                    until: None,
                    backoff: Duration::ZERO,
                });
                Err(message)
            }
            Err(e) => {
                let backoff = match &state.failed {
                    Some(f) if f.until.is_some() => (f.backoff * 2).min(MAX_BACKOFF),
                    _ => self.first_backoff,
                };
                let message = format!(
                    "mcp server `{}` rejected its OAuth token and the refresh failed ({}); it will \
                     be tried again in {}s",
                    self.server,
                    e.reason,
                    backoff.as_secs()
                );
                tracing::warn!(server = %self.server, error = %e.reason, ?backoff, "MCP OAuth refresh failed");
                state.failed = Some(Failed {
                    message: message.clone(),
                    until: Some(now + backoff),
                    backoff,
                });
                Err(message)
            }
        }
    }
}

/// A currently-valid bearer token from `name`'s stored login, if one exists — read and, when it is
/// about to expire, refreshed (and re-persisted) by rmcp's
/// [`AuthorizationManager::get_access_token`](rmcp::transport::auth::AuthorizationManager::get_access_token).
/// `None` for every case short of a token to attach: the caller connects unauthenticated and lets
/// the server's answer be the signal.
async fn stored_token(name: &str, url: &str) -> Option<String> {
    let manager = match manager(name, url).await {
        Ok(manager) => manager,
        Err(e) => {
            tracing::warn!(server = %name, error = %e.reason, "failed to restore a stored MCP OAuth login");
            return None;
        }
    };
    match manager.get_access_token().await {
        Ok(token) => Some(token),
        Err(e) => {
            tracing::warn!(
                server = %name,
                error = %e,
                "stored MCP OAuth login could not be refreshed; run `agent mcp-login {name}` again"
            );
            None
        }
    }
}

/// An `AuthorizationManager` for `name` at `url`, restored from the `mcp-login` store.
async fn manager(
    name: &str,
    url: &str,
) -> Result<rmcp::transport::auth::AuthorizationManager, RefreshError> {
    let mut manager = rmcp::transport::auth::AuthorizationManager::new(url)
        .await
        .map_err(RefreshError::from_auth)?;
    manager.set_credential_store(crate::mcp_auth_store::McpAuthStore::open_default().scoped(name));
    match manager.initialize_from_store().await {
        Ok(true) => Ok(manager),
        Ok(false) => Err(RefreshError::definitive("no stored login")),
        Err(e) => Err(RefreshError::from_auth(e)),
    }
}

/// Refresh `name`'s token regardless of its recorded expiry (the server just rejected it), and
/// persist it. Returns the new access token.
async fn refresh(name: &str, url: &str) -> Result<String, RefreshError> {
    let response = manager(name, url)
        .await?
        .refresh_token()
        .await
        .map_err(RefreshError::from_auth)?;
    // `OAuthTokenResponse`'s accessor trait lives in `oauth2`, which this crate does not name;
    // its serialized form is the RFC 6749 token response.
    serde_json::to_value(&response)
        .ok()
        .and_then(|v| v.get("access_token")?.as_str().map(str::to_owned))
        .ok_or_else(|| RefreshError::transient("the token response carried no access_token"))
}

/// Whether an error from the streamable-HTTP client is the server's 401 — with or without a
/// `WWW-Authenticate` challenge (only one carrying it becomes rmcp's `AuthRequired`; a bare one is
/// rmcp's `"HTTP 401 …"` response error, or a status error on the standalone stream).
fn is_unauthorized(e: &StreamableHttpError<reqwest::Error>) -> bool {
    match e {
        StreamableHttpError::AuthRequired(_) => true,
        StreamableHttpError::Client(e) => e.status() == Some(reqwest::StatusCode::UNAUTHORIZED),
        StreamableHttpError::UnexpectedServerResponse(m) => m.starts_with("HTTP 401"),
        _ => false,
    }
}

/// A streamable-HTTP client whose requests carry the server's current OAuth token and, on a 401,
/// refresh it and retry once. Without a [`ServerAuth`] (no login, or a grant connector whose
/// credentials are its own headers) it is the inner client unchanged. See the module doc.
#[derive(Clone)]
pub(crate) struct OAuthHttp<C> {
    inner: C,
    auth: Option<Arc<ServerAuth>>,
}

impl<C> OAuthHttp<C> {
    pub(crate) fn new(inner: C, auth: Option<Arc<ServerAuth>>) -> Self {
        Self { inner, auth }
    }
}

impl<C: StreamableHttpClient<Error = reqwest::Error>> OAuthHttp<C> {
    async fn authed<T, F, Fut>(
        &self,
        passed: Option<String>,
        call: F,
    ) -> Result<T, StreamableHttpError<C::Error>>
    where
        F: Fn(Option<String>) -> Fut,
        Fut: Future<Output = Result<T, StreamableHttpError<C::Error>>>,
    {
        let Some(auth) = &self.auth else {
            return call(passed).await;
        };
        let sent = auth.token().await;
        match call(sent.clone()).await {
            Err(e) if is_unauthorized(&e) => {
                match auth.after_rejection(sent.as_deref()).await {
                    // Once: a second 401 is the server's answer, returned as is.
                    Ok(fresh) => call(Some(fresh)).await,
                    Err(message) => Err(StreamableHttpError::UnexpectedServerResponse(
                        message.into(),
                    )),
                }
            }
            other => other,
        }
    }
}

impl<C> StreamableHttpClient for OAuthHttp<C>
where
    C: StreamableHttpClient<Error = reqwest::Error> + Send + Sync,
{
    type Error = C::Error;

    async fn post_message(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
        self.authed(auth_header, |token| {
            self.inner.post_message(
                uri.clone(),
                message.clone(),
                session_id.clone(),
                token,
                custom_headers.clone(),
            )
        })
        .await
    }

    async fn post_message_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
        self.authed(auth_header, |token| {
            self.inner.post_message_with_max_sse_event_size(
                uri.clone(),
                message.clone(),
                session_id.clone(),
                token,
                custom_headers.clone(),
                max_sse_event_size,
            )
        })
        .await
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session_id: Arc<str>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<(), StreamableHttpError<Self::Error>> {
        self.authed(auth_header, |token| {
            self.inner.delete_session(
                uri.clone(),
                session_id.clone(),
                token,
                custom_headers.clone(),
            )
        })
        .await
    }

    async fn get_stream(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<
        BoxStream<'static, Result<sse_stream::Sse, SseError>>,
        StreamableHttpError<Self::Error>,
    > {
        self.authed(auth_header, |token| {
            self.inner.get_stream(
                uri.clone(),
                session_id.clone(),
                last_event_id.clone(),
                token,
                custom_headers.clone(),
            )
        })
        .await
    }

    async fn get_stream_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<
        BoxStream<'static, Result<sse_stream::Sse, SseError>>,
        StreamableHttpError<Self::Error>,
    > {
        self.authed(auth_header, |token| {
            self.inner.get_stream_with_max_sse_event_size(
                uri.clone(),
                session_id.clone(),
                last_event_id.clone(),
                token,
                custom_headers.clone(),
                max_sse_event_size,
            )
        })
        .await
    }
}

#[cfg(test)]
impl ServerAuth {
    /// A `ServerAuth` holding `token`, whose refresh is `refresh` instead of the authorization
    /// server.
    pub(crate) fn fake(
        token: &str,
        refresh: impl Fn() -> Result<String, RefreshError> + Send + Sync + 'static,
    ) -> Arc<Self> {
        let mut auth = Self::new("fake", "http://127.0.0.1:9/mcp");
        auth.state = tokio::sync::Mutex::new(TokenState {
            token: Some(token.into()),
            failed: None,
        });
        auth.fake_refresh = Some(Arc::new(refresh));
        Arc::new(auth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn concurrent_rejections_of_one_token_share_one_refresh() {
        let refreshes = Arc::new(AtomicU32::new(0));
        let counter = refreshes.clone();
        let auth = ServerAuth::fake("old", move || {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            Ok(format!("new-{n}"))
        });
        let got =
            futures::future::join_all((0..8).map(|_| auth.after_rejection(Some("old")))).await;
        assert!(got.iter().all(|t| t.as_deref() == Ok("new-0")), "{got:?}");
        assert_eq!(refreshes.load(Ordering::SeqCst), 1);
        // A rejection of the *new* token is a new event: one more refresh.
        assert_eq!(
            auth.after_rejection(Some("new-0")).await.as_deref(),
            Ok("new-1")
        );
        assert_eq!(auth.token().await.as_deref(), Some("new-1"));
    }

    #[tokio::test]
    async fn a_definitive_refresh_failure_names_mcp_login_and_is_remembered_for_good() {
        let refreshes = Arc::new(AtomicU32::new(0));
        let counter = refreshes.clone();
        let mut auth = ServerAuth::fake("old", move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Err(RefreshError::definitive("invalid_grant"))
        });
        Arc::get_mut(&mut auth).unwrap().first_backoff = Duration::from_millis(10);
        for _ in 0..3 {
            let e = auth.after_rejection(Some("old")).await.unwrap_err();
            assert!(e.contains("agent mcp-login fake"), "{e}");
            // Even past any backoff, a rejected refresh token is not tried again.
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
        assert_eq!(refreshes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_transient_refresh_failure_backs_off_doubling_then_refreshes_again() {
        let refreshes = Arc::new(AtomicU32::new(0));
        let counter = refreshes.clone();
        // Fails twice (the authorization server is down), then works.
        let mut auth = ServerAuth::fake("old", move || {
            match counter.fetch_add(1, Ordering::SeqCst) {
                0 | 1 => Err(RefreshError::transient("connection refused")),
                _ => Ok("fresh".into()),
            }
        });
        Arc::get_mut(&mut auth).unwrap().first_backoff = Duration::from_millis(200);
        let backoff = |auth: &ServerAuth| {
            auth.state
                .try_lock()
                .unwrap()
                .failed
                .as_ref()
                .map(|f| f.backoff)
        };

        let e = auth.after_rejection(Some("old")).await.unwrap_err();
        assert!(
            !e.contains("mcp-login"),
            "a transient failure is not a login problem: {e}"
        );
        assert_eq!(backoff(&auth), Some(Duration::from_millis(200)));
        // Within the backoff: remembered, the authorization server is not asked.
        auth.after_rejection(Some("old")).await.unwrap_err();
        assert_eq!(refreshes.load(Ordering::SeqCst), 1);

        // Past it: refreshed again; it fails again, and the backoff doubles.
        tokio::time::sleep(Duration::from_millis(250)).await;
        auth.after_rejection(Some("old")).await.unwrap_err();
        assert_eq!(refreshes.load(Ordering::SeqCst), 2);
        assert_eq!(backoff(&auth), Some(Duration::from_millis(400)));
        tokio::time::sleep(Duration::from_millis(250)).await;
        auth.after_rejection(Some("old")).await.unwrap_err();
        assert_eq!(
            refreshes.load(Ordering::SeqCst),
            2,
            "still inside the doubled backoff"
        );

        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            auth.after_rejection(Some("old")).await.as_deref(),
            Ok("fresh")
        );
        assert_eq!(refreshes.load(Ordering::SeqCst), 3);
        assert_eq!(backoff(&auth), None, "a success clears the failure");
    }

    #[test]
    fn the_backoff_is_capped() {
        let mut b = FIRST_BACKOFF;
        for _ in 0..10 {
            b = (b * 2).min(MAX_BACKOFF);
        }
        assert_eq!(b, MAX_BACKOFF);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_reload_racing_a_refresh_never_puts_back_the_older_token() {
        // The store, as the refresh persists into it and a reload reads it.
        let store = Arc::new(Mutex::new("old".to_owned()));
        let persist = store.clone();
        let mut auth = ServerAuth::fake("old", move || {
            // A slow token endpoint: the reload below starts while this is in flight.
            std::thread::sleep(Duration::from_millis(300));
            *persist.lock().unwrap() = "fresh".into();
            Ok("fresh".into())
        });
        let read = store.clone();
        Arc::get_mut(&mut auth).unwrap().fake_stored =
            Some(Arc::new(move || Some(read.lock().unwrap().clone())));

        let refreshing = {
            let auth = auth.clone();
            tokio::spawn(async move { auth.after_rejection(Some("old")).await })
        };
        tokio::time::sleep(Duration::from_millis(100)).await;
        auth.reload().await;
        assert_eq!(refreshing.await.unwrap().as_deref(), Ok("fresh"));
        assert_eq!(
            auth.token().await.as_deref(),
            Some("fresh"),
            "a redial's reload must not clobber the token a concurrent refresh just set"
        );
    }

    #[tokio::test]
    async fn a_reload_that_reads_nothing_keeps_the_token_held() {
        let mut auth = ServerAuth::fake("fresh", || Ok("unused".into()));
        Arc::get_mut(&mut auth).unwrap().fake_stored = Some(Arc::new(|| None));
        auth.reload().await;
        assert_eq!(auth.token().await.as_deref(), Some("fresh"));
    }

    /// A loopback MCP endpoint: `Bearer fresh` gets a `tools/list` result; any other token gets
    /// `status` (with `extra` headers). Records each request's `Authorization`.
    async fn endpoint(
        status: &'static str,
        extra: &'static str,
    ) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let record = seen.clone();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let record = record.clone();
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
                    let mut buf = vec![0u8; 16384];
                    let n = stream.read(&mut buf).await.unwrap_or(0);
                    let head = String::from_utf8_lossy(&buf[..n]).to_ascii_lowercase();
                    let auth = head
                        .lines()
                        .find_map(|l| l.strip_prefix("authorization: "))
                        .unwrap_or_default()
                        .to_owned();
                    record.lock().unwrap().push(auth.clone());
                    let reply = if auth == "bearer fresh" {
                        let body = r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}"#;
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                    } else {
                        format!(
                            "HTTP/1.1 {status}\r\n{extra}Content-Length: 0\r\nConnection: close\r\n\r\n"
                        )
                    };
                    let _ = stream.write_all(reply.as_bytes()).await;
                });
            }
        });
        (url, seen)
    }

    fn client(auth: Arc<ServerAuth>) -> OAuthHttp<crate::tools::mcp_view_http::ViewCappedHttp> {
        agent_core::ensure_provider();
        OAuthHttp::new(
            crate::tools::mcp_view_http::ViewCappedHttp::new(crate::tools::mcp_wire::HttpClient(
                reqwest::Client::new(),
            )),
            Some(auth),
        )
    }

    fn tools_list() -> ClientJsonRpcMessage {
        serde_json::from_value(
            serde_json::json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {} }),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn a_bare_401_without_a_challenge_also_refreshes_and_retries() {
        // Same rule as the direct `events/*` path: any 401 from a server with a login refreshes.
        let (url, seen) = endpoint("401 Unauthorized", "").await;
        let auth = ServerAuth::fake("stale", || Ok("fresh".into()));
        let response = client(auth)
            .post_message(url.into(), tools_list(), None, None, HashMap::new())
            .await
            .unwrap();
        assert!(
            matches!(response, StreamableHttpPostResponse::Json(..)),
            "{response:?}"
        );
        assert_eq!(*seen.lock().unwrap(), ["bearer stale", "bearer fresh"]);
    }

    #[tokio::test]
    async fn a_403_insufficient_scope_passes_through_without_a_refresh() {
        let (url, seen) = endpoint(
            "403 Forbidden",
            "WWW-Authenticate: Bearer error=\"insufficient_scope\", scope=\"admin\"\r\n",
        )
        .await;
        let refreshes = Arc::new(AtomicU32::new(0));
        let counter = refreshes.clone();
        let auth = ServerAuth::fake("stale", move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok("fresh".into())
        });
        let e = client(auth)
            .post_message(url.into(), tools_list(), None, None, HashMap::new())
            .await
            .unwrap_err();
        assert!(
            matches!(e, StreamableHttpError::InsufficientScope(_)),
            "{e:?}"
        );
        assert_eq!(refreshes.load(Ordering::SeqCst), 0, "a 403 never refreshes");
        assert_eq!(seen.lock().unwrap().len(), 1, "nor is it retried");
    }
}
