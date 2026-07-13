//! A one-shot local HTTP listener for the PKCE authorization-code callback — Anthropic's fixed port
//! `53692`, OpenAI Codex's browser flow on fixed port `1455`. Both are bound to a fixed port because
//! that's the literal `redirect_uri` each provider's OAuth client was registered against; there is no
//! dynamic-port variant.
//!
//! Built on `tiny_http` rather than `hyper`/`axum` (no `Service`/executor wiring needed for "accept
//! exactly one real request, answer it, stop") or `pingora` (a full L7 proxy, wildly overkill for a
//! loopback listener that lives for one request). `tiny_http::Server` is synchronous, so it's bridged
//! into tokio via `spawn_blocking`, polled against a cancellation token every [`POLL_INTERVAL`].

use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use super::error::OAuthError;

/// How often the blocking accept loop wakes up to check the cancellation token if `unblock()` (fired
/// the instant cancellation happens) doesn't get there first — a latency ceiling, not a correctness
/// concern.
const POLL_INTERVAL: Duration = Duration::from_millis(250);

pub struct CallbackServer {
    server: Arc<tiny_http::Server>,
}

impl CallbackServer {
    /// Bind `host:port` for exactly one callback. Returns [`OAuthError::PortBindFailed`] if the port
    /// is already in use — callers decide what that means: Anthropic propagates it as a hard error
    /// (matching pi, which has no fallback there either); OpenAI Codex's browser flow falls back to
    /// manual-paste-only (see `openai_codex.rs`).
    pub fn bind(host: &str, port: u16) -> Result<Self, OAuthError> {
        let server =
            tiny_http::Server::http((host, port)).map_err(|source| OAuthError::PortBindFailed {
                host: host.to_string(),
                port,
                detail: source.to_string(),
            })?;
        Ok(Self {
            server: Arc::new(server),
        })
    }

    /// Waits for exactly one request on `expected_path` whose `state` query param equals
    /// `expected_state` and which carries a `code`. A request that's on the wrong path, missing
    /// `code`/`state`, carries an `error` param, or has a mismatched `state` gets an HTML error page
    /// but does **not** resolve the wait — the server keeps listening. This matches both Anthropic's
    /// and OpenAI Codex's real callback handlers, which retry silently on a bad request rather than
    /// failing the whole login over one malformed hit (a stray retry, a browser prefetch, a scanner).
    /// Returns `None` only on cancellation.
    pub async fn wait_for_callback(
        &self,
        expected_path: &'static str,
        expected_state: String,
        cancel: CancellationToken,
    ) -> Option<String> {
        let server = self.server.clone();
        let blocking_cancel = cancel.clone();

        // Nudges the blocking accept loop awake the instant cancellation fires, rather than making it
        // wait out its own poll tick — a latency nicety, not a correctness requirement (the loop below
        // also checks `is_cancelled()` on every tick regardless).
        // Aborted on *every* exit from this function, including an early drop of the whole future — not
        // just the normal return. It holds its own `Arc<Server>`, so a task left parked on
        // `cancel.cancelled()` for a token that never fires would keep the listener (and its bound,
        // fixed port) alive no matter what the accept loop below does.
        let unblocker = AbortOnDrop(Some({
            let server = server.clone();
            tokio::spawn(async move {
                cancel.cancelled().await;
                server.unblock();
            })
        }));

        let result = tokio::task::spawn_blocking(move || {
            loop {
                if blocking_cancel.is_cancelled() {
                    return None;
                }
                match server.recv_timeout(POLL_INTERVAL) {
                    Ok(Some(request)) => {
                        match handle_one(&request, expected_path, &expected_state) {
                            Some(code) => {
                                respond(request, true);
                                return Some(code);
                            }
                            None => {
                                respond(request, false);
                                // keep listening — do not resolve on a bad request.
                            }
                        }
                    }
                    Ok(None) => continue,  // poll tick, nothing yet
                    Err(_) => return None, // socket error, or unblock()ed with nothing pending
                }
            }
        })
        .await
        .ok()
        .flatten();

        drop(unblocker);
        result
    }
}

/// Aborts a task when it goes out of scope, however that happens — a normal return, an early `?`, or
/// the enclosing future being dropped mid-`.await`. A bare `handle.abort()` at the end of a function
/// only covers the first of those.
struct AbortOnDrop(Option<tokio::task::JoinHandle<()>>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if let Some(h) = self.0.take() {
            h.abort();
        }
    }
}

impl Drop for CallbackServer {
    /// Releases the listener — and with it the bound port — without relying on anyone to have cancelled
    /// the token first.
    ///
    /// The accept loop runs on `spawn_blocking`, which cannot be aborted: it exits only by observing a
    /// cancelled token or by `recv_timeout` failing. So if a caller merely *drops* the login future
    /// rather than cancelling it, that loop keeps polling forever, holding its own `Arc<Server>` and the
    /// port with it. `unblock()` makes the pending `recv_timeout` fail, the loop returns, and its `Arc`
    /// goes with it. This matters more than it looks: the port is *fixed* per provider (53692 for
    /// Anthropic, 1455 for Codex), not ephemeral, so a single leaked listener makes every subsequent
    /// login on that provider fail outright with `PortBindFailed`.
    fn drop(&mut self) {
        self.server.unblock();
    }
}

/// Parse one request; `Some(code)` only when the path/state/code all check out. Deliberately
/// returns `Option`, not `Result` — see [`CallbackServer::wait_for_callback`]'s doc comment on why a
/// bad request must not fail the wait.
fn handle_one(
    request: &tiny_http::Request,
    expected_path: &str,
    expected_state: &str,
) -> Option<String> {
    let url = url::Url::parse(&format!("http://localhost{}", request.url())).ok()?;
    if url.path() != expected_path {
        return None;
    }
    let mut code = None;
    let mut state = None;
    let mut saw_error = false;
    for (k, v) in url.query_pairs() {
        match &*k {
            "code" => code = Some(v.into_owned()),
            "state" => state = Some(v.into_owned()),
            "error" => saw_error = true,
            _ => {}
        }
    }
    if saw_error {
        return None;
    }
    let (code, state) = (code?, state?);
    if state != expected_state {
        return None;
    }
    Some(code)
}

fn respond(request: tiny_http::Request, success: bool) {
    let (status, body) = if success {
        (
            200,
            "<html><body>Login complete \u{2014} you can close this tab.</body></html>",
        )
    } else {
        (
            400,
            "<html><body>Login failed \u{2014} the request was missing required parameters or had \
             an invalid state. You can close this tab and retry.</body></html>",
        )
    };
    #[allow(clippy::expect_used)] // static header name/value; construction cannot fail
    let header =
        tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..])
            .expect("static header is valid");
    let response = tiny_http::Response::from_string(body)
        .with_status_code(status)
        .with_header(header);
    let _ = request.respond(response);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn free_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    #[tokio::test]
    async fn a_matching_callback_resolves_with_the_code() {
        let port = free_port();
        let server = CallbackServer::bind("127.0.0.1", port).unwrap();
        let client = tokio::spawn(async move {
            // Give the server a moment to be actively waiting.
            tokio::time::sleep(Duration::from_millis(50)).await;
            reqwest::Client::new()
                .get(format!(
                    "http://127.0.0.1:{port}/callback?code=abc123&state=xyz"
                ))
                .send()
                .await
                .unwrap()
        });
        let code = server
            .wait_for_callback("/callback", "xyz".to_string(), CancellationToken::new())
            .await;
        let resp = client.await.unwrap();
        assert!(resp.status().is_success());
        assert_eq!(code, Some("abc123".to_string()));
    }

    #[tokio::test]
    async fn a_state_mismatch_does_not_resolve_and_the_server_keeps_listening() {
        let port = free_port();
        let server = CallbackServer::bind("127.0.0.1", port).unwrap();
        let cancel = CancellationToken::new();
        let cancel_for_client = cancel.clone();
        let client = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let http = reqwest::Client::new();
            let bad = http
                .get(format!(
                    "http://127.0.0.1:{port}/callback?code=abc&state=WRONG"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(bad.status().as_u16(), 400);
            // The server must still be listening after the bad request — a good one now succeeds.
            let good = http
                .get(format!(
                    "http://127.0.0.1:{port}/callback?code=abc&state=right"
                ))
                .send()
                .await
                .unwrap();
            assert!(good.status().is_success());
            cancel_for_client.cancel(); // unused here, just documents intent if the wait ever hangs
        });
        let code = server
            .wait_for_callback("/callback", "right".to_string(), cancel)
            .await;
        client.await.unwrap();
        assert_eq!(code, Some("abc".to_string()));
    }

    #[tokio::test]
    async fn cancellation_resolves_with_none() {
        let port = free_port();
        let server = CallbackServer::bind("127.0.0.1", port).unwrap();
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            cancel_clone.cancel();
        });
        let code = server
            .wait_for_callback("/callback", "irrelevant".to_string(), cancel)
            .await;
        assert_eq!(code, None);
    }

    #[tokio::test]
    async fn binding_an_already_used_port_fails() {
        let port = free_port();
        let _first = CallbackServer::bind("127.0.0.1", port).unwrap();
        let second = CallbackServer::bind("127.0.0.1", port);
        assert!(matches!(second, Err(OAuthError::PortBindFailed { .. })));
    }
}
