//! Mid-session MCP OAuth: a server that answers **401** to a request (its token expired, or was
//! revoked, while the session was running) gets one token refresh and one retry, instead of failing
//! every request until the server is redialed.
//!
//! - [`ServerAuth`] is one configured server's bearer token, shared by **every** connection to it in
//!   this process (the plain and the apps-flavoured connections, and direct `events/*` requests).
//!   It is loaded from the `agent mcp-login` store at each dial, as before.
//! - On a 401, [`ServerAuth::after_rejection`] refreshes through the same
//!   [`AuthorizationManager`](rmcp::transport::auth::AuthorizationManager) and
//!   [`McpAuthStore`](crate::mcp_auth_store::McpAuthStore) `mcp-login` uses — so the refreshed token
//!   is persisted, and the next process starts from it. **Single flight:** the refresh runs under the
//!   token's lock, and a caller whose rejected token is no longer current takes the new one without
//!   refreshing again, so any number of concurrent 401s cost one refresh (which also matters for an
//!   authorization server that rotates refresh tokens: a second refresh with the spent one would
//!   fail).
//! - [`OAuthHttp`] applies that to every request rmcp's streamable-HTTP transport makes —
//!   `tools/call`, `resources/*`, `prompts/*`, `skills/*`, MCP App view reads, the handshake, the
//!   standalone SSE stream — retrying each **once**. A 403 (`InsufficientScope`) passes through
//!   untouched: a refresh does not widen scopes. A refresh that fails surfaces as an error naming
//!   `agent mcp-login <server>`, and is not attempted again for the same rejected token.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};

use futures::stream::BoxStream;
use http::{HeaderName, HeaderValue};
use rmcp::model::ClientJsonRpcMessage;
use rmcp::transport::streamable_http_client::{
    SseError, StreamableHttpClient, StreamableHttpError, StreamableHttpPostResponse,
};

/// One configured server's OAuth bearer token, shared by every connection to it. See the module doc.
pub(crate) struct ServerAuth {
    server: String,
    url: String,
    state: tokio::sync::Mutex<TokenState>,
    /// A stand-in for the authorization server, for unit tests (`None`: the real refresh).
    #[cfg(test)]
    fake_refresh: Option<Arc<dyn Fn() -> Result<String, String> + Send + Sync>>,
}

#[derive(Default)]
struct TokenState {
    /// What every request to the server sends; `None` when the stored login yields no usable token.
    token: Option<String>,
    /// Set when a refresh after `token` was rejected failed: the error every later rejection of the
    /// same token returns, without asking the authorization server again.
    failed: Option<String>,
}

/// Every server's [`ServerAuth`] in this process, by `(name, url)`.
static SERVERS: Mutex<Vec<Arc<ServerAuth>>> = Mutex::new(Vec::new());

impl ServerAuth {
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
                    let auth = Arc::new(Self {
                        server: name.to_owned(),
                        url: url.to_owned(),
                        state: tokio::sync::Mutex::new(TokenState::default()),
                        #[cfg(test)]
                        fake_refresh: None,
                    });
                    servers.push(auth.clone());
                    auth
                }
            }
        };
        let token = stored_token(name, url).await;
        let mut state = auth.state.lock().await;
        // A login (or logout) made since the last dial takes effect here, as it always did.
        if state.token != token {
            state.failed = None;
        }
        state.token = token;
        drop(state);
        Some(auth)
    }

    /// The token requests send now.
    pub(crate) async fn token(&self) -> Option<String> {
        self.state.lock().await.token.clone()
    }

    /// `sent` was answered 401: the token to retry with. If another caller already replaced `sent`,
    /// that replacement — no second refresh. Otherwise one refresh, persisted; its failure is the
    /// `Err` (naming `agent mcp-login`), remembered for `sent`.
    pub(crate) async fn after_rejection(&self, sent: Option<&str>) -> Result<String, String> {
        let mut state = self.state.lock().await;
        if let Some(current) = state.token.as_deref()
            && Some(current) != sent
        {
            return Ok(current.to_owned());
        }
        if let Some(failed) = &state.failed {
            return Err(failed.clone());
        }
        #[cfg(test)]
        let refreshed = match &self.fake_refresh {
            Some(fake) => fake(),
            None => refresh(&self.server, &self.url).await,
        };
        #[cfg(not(test))]
        let refreshed = refresh(&self.server, &self.url).await;
        match refreshed {
            Ok(token) => {
                tracing::info!(server = %self.server, "refreshed a rejected MCP OAuth token");
                state.token = Some(token.clone());
                Ok(token)
            }
            Err(e) => {
                let message = format!(
                    "mcp server `{0}` rejected its OAuth token and it could not be refreshed ({e}); \
                     run `agent mcp-login {0}` again",
                    self.server
                );
                tracing::warn!(server = %self.server, error = %e, "MCP OAuth refresh failed");
                state.failed = Some(message.clone());
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
            tracing::warn!(server = %name, error = %e, "failed to restore a stored MCP OAuth login");
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
) -> Result<rmcp::transport::auth::AuthorizationManager, String> {
    let mut manager = rmcp::transport::auth::AuthorizationManager::new(url)
        .await
        .map_err(|e| e.to_string())?;
    manager.set_credential_store(crate::mcp_auth_store::McpAuthStore::open_default().scoped(name));
    match manager.initialize_from_store().await {
        Ok(true) => Ok(manager),
        Ok(false) => Err("no stored login".into()),
        Err(e) => Err(e.to_string()),
    }
}

/// Refresh `name`'s token regardless of its recorded expiry (the server just rejected it), and
/// persist it. Returns the new access token.
async fn refresh(name: &str, url: &str) -> Result<String, String> {
    let response = manager(name, url)
        .await?
        .refresh_token()
        .await
        .map_err(|e| e.to_string())?;
    // `OAuthTokenResponse`'s accessor trait lives in `oauth2`, which this crate does not name;
    // its serialized form is the RFC 6749 token response.
    serde_json::to_value(&response)
        .ok()
        .and_then(|v| v.get("access_token")?.as_str().map(str::to_owned))
        .ok_or_else(|| "the token response carried no access_token".into())
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

impl<C: StreamableHttpClient> OAuthHttp<C> {
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
            Err(StreamableHttpError::AuthRequired(_)) => {
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
    C: StreamableHttpClient + Send + Sync,
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
        refresh: impl Fn() -> Result<String, String> + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            server: "fake".into(),
            url: "http://127.0.0.1:9/mcp".into(),
            state: tokio::sync::Mutex::new(TokenState {
                token: Some(token.into()),
                failed: None,
            }),
            fake_refresh: Some(Arc::new(refresh)),
        })
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
    async fn a_failed_refresh_names_mcp_login_and_is_not_repeated_for_the_same_token() {
        let refreshes = Arc::new(AtomicU32::new(0));
        let counter = refreshes.clone();
        let auth = ServerAuth::fake("old", move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Err("invalid_grant".into())
        });
        for _ in 0..3 {
            let e = auth.after_rejection(Some("old")).await.unwrap_err();
            assert!(e.contains("agent mcp-login fake"), "{e}");
        }
        assert_eq!(refreshes.load(Ordering::SeqCst), 1);
    }
}
