//! Ending streamable-HTTP MCP sessions on the way out of the process.
//!
//! A server that issued an `Mcp-Session-Id` keeps that session until it is told it is over — the
//! spec's `DELETE` — or until it times out on its own. rmcp sends that `DELETE` when a connection is
//! closed while the runtime is running (the idle reaper, a registry rebuild); a process that simply
//! exits (`agent run` finishing, a `serve` shutdown) never got that far, and every session it held
//! was left for the server to expire.
//!
//! So each connection records the session it opened in an [`HttpSession`] (kept by
//! [`OAuthHttp`](crate::tools::mcp_oauth::OAuthHttp), which sees every response's session id and
//! every `DELETE` rmcp sends), and [`begin_close_all`] — run beside the stdio sweep in
//! `mcp_stdio::sweep_before_exit` — sends a `DELETE` for every session still open: in parallel, with
//! the server's bearer token and the connection's own headers, under one bounded deadline.
//!
//! It runs on a thread of its own with a fresh runtime and a fresh, pool-less client: the process
//! runtime may already be gone (`main`'s last guard) or blocked in the very call that is exiting,
//! and a pooled connection belonging to a blocked runtime would never answer. A grant connector's
//! client gets the same SSRF resolver its connection used. A server that does not answer costs at
//! most the deadline; the session then expires on the server's side, as before.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use http::{HeaderName, HeaderValue};

/// The most the exit waits for every `DELETE` together.
pub const CLOSE_DEADLINE: Duration = Duration::from_millis(1500);

/// Headers a `DELETE` never takes from the connection's (rmcp's reserved set, plus the two it sets).
const NOT_COPIED: [&str; 5] = [
    "accept",
    "mcp-session-id",
    "last-event-id",
    "authorization",
    "content-type",
];

/// One HTTP connection's open session, if it has one. See the module doc.
///
/// Held by the connection's transport (`OAuthHttp`) and only **weakly** by the process: a connection
/// that goes away takes its entry with it, so a long-lived daemon never accumulates them. If it goes
/// away with its session still open (its transport died before rmcp's own `DELETE`), dropping it
/// sends that `DELETE` right away on the runtime it was dropped on, and keeps it as an [`Ending`]
/// for the exit only until that succeeds — or, during the runtime's own teardown, where nothing
/// spawned runs any more, for the exit to send.
pub(crate) struct HttpSession {
    registry: Arc<Registry>,
    uri: String,
    auth: Option<Arc<crate::tools::mcp_oauth::ServerAuth>>,
    /// A grant connector's egress policy: its exit client resolves through the same SSRF resolver.
    egress: Option<Arc<crate::tools::web::ssrf::EgressPolicy>>,
    open: Mutex<Option<Open>>,
}

struct Open {
    id: String,
    headers: HashMap<HeaderName, HeaderValue>,
}

/// One session's `DELETE`, everything needed to send it.
#[derive(Clone)]
struct Ending {
    uri: String,
    id: String,
    headers: HashMap<HeaderName, HeaderValue>,
    auth: Option<Arc<crate::tools::mcp_oauth::ServerAuth>>,
    egress: Option<Arc<crate::tools::web::ssrf::EgressPolicy>>,
}

/// Where the exit finds what it has to end. One per process ([`Registry::process`]); a test makes
/// its own, so tests never see each other's sessions.
pub(crate) struct Registry {
    /// Connections with a session open, weakly.
    open: Mutex<Vec<std::sync::Weak<HttpSession>>>,
    /// Sessions whose connection went away still open, by a sequence number, while their `DELETE`
    /// is in flight — or, if their runtime was already tearing down, until the exit sends it.
    orphans: Mutex<Vec<(u64, Ending)>>,
    /// How long an orphan's `DELETE` may take before it is given up.
    orphan_timeout: Duration,
}

impl Default for Registry {
    fn default() -> Self {
        Self {
            open: Mutex::default(),
            orphans: Mutex::default(),
            orphan_timeout: CLOSE_DEADLINE * 4,
        }
    }
}

impl Registry {
    pub(crate) fn process() -> Arc<Registry> {
        static PROCESS: std::sync::LazyLock<Arc<Registry>> = std::sync::LazyLock::new(Arc::default);
        PROCESS.clone()
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl HttpSession {
    pub(crate) fn new(
        uri: &str,
        auth: Option<Arc<crate::tools::mcp_oauth::ServerAuth>>,
        egress: Option<Arc<crate::tools::web::ssrf::EgressPolicy>>,
    ) -> Arc<Self> {
        Self::new_in(Registry::process(), uri, auth, egress)
    }

    pub(crate) fn new_in(
        registry: Arc<Registry>,
        uri: &str,
        auth: Option<Arc<crate::tools::mcp_oauth::ServerAuth>>,
        egress: Option<Arc<crate::tools::web::ssrf::EgressPolicy>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            registry,
            uri: uri.to_owned(),
            auth,
            egress,
            open: Mutex::new(None),
        })
    }

    /// A response named the session `id` (the handshake's, or a re-initialized one's); `headers`
    /// are the request's own, which a `DELETE` for it carries too.
    pub(crate) fn opened(self: &Arc<Self>, id: &str, headers: &HashMap<HeaderName, HeaderValue>) {
        let mut open = lock(&self.open);
        if open.as_ref().is_some_and(|o| o.id == id) {
            return;
        }
        *open = Some(Open {
            id: id.to_owned(),
            headers: headers.clone(),
        });
        drop(open);
        let mut all = lock(&self.registry.open);
        all.retain(|s| s.strong_count() > 0);
        if !all
            .iter()
            .any(|s| std::ptr::eq(s.as_ptr(), Arc::as_ptr(self)))
        {
            all.push(Arc::downgrade(self));
        }
    }

    /// rmcp is ending the session `id` itself (its own `DELETE`): nothing left for the exit to do.
    pub(crate) fn closed(self: &Arc<Self>, id: &str) {
        let mut open = lock(&self.open);
        if open.as_ref().is_some_and(|o| o.id == id) {
            *open = None;
            drop(open);
            lock(&self.registry.open)
                .retain(|s| s.strong_count() > 0 && !std::ptr::eq(s.as_ptr(), Arc::as_ptr(self)));
        }
    }

    fn ending(&self) -> Option<Ending> {
        let open = lock(&self.open);
        let open = open.as_ref()?;
        Some(Ending {
            uri: self.uri.clone(),
            id: open.id.clone(),
            headers: open.headers.clone(),
            auth: self.auth.clone(),
            egress: self.egress.clone(),
        })
    }
}

impl Drop for HttpSession {
    fn drop(&mut self) {
        let Some(ending) = self.ending() else {
            return;
        };
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        lock(&self.registry.orphans).push((n, ending.clone()));
        // Send it now if a runtime is here to run it. Once that attempt is over — answered, refused
        // or timed out — the entry goes: an unanswered `DELETE` is not kept for the process's life.
        // Only a drop during the runtime's own teardown, where nothing spawned runs any more, leaves
        // it for the exit to send.
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let registry = self.registry.clone();
            handle.spawn(async move {
                let _ = tokio::time::timeout(registry.orphan_timeout, ending.send()).await;
                lock(&registry.orphans).retain(|(m, _)| *m != n);
            });
        }
    }
}

impl Ending {
    /// Send this session's `DELETE`: the status, or why it was not sent.
    async fn send(&self) -> Result<u16, String> {
        let client = self.client()?;
        let mut request = client
            .delete(&self.uri)
            .header("Mcp-Session-Id", self.id.as_str());
        for (name, value) in &self.headers {
            if !NOT_COPIED.contains(&name.as_str()) {
                request = request.header(name, value);
            }
        }
        if let Some(auth) = &self.auth
            && let Some(token) = auth.token().await
        {
            request = request.bearer_auth(token);
        }
        request
            .send()
            .await
            .map(|r| r.status().as_u16())
            .map_err(|e| e.to_string())
    }

    fn client(&self) -> Result<reqwest::Client, String> {
        agent_core::ensure_provider();
        let mut builder = reqwest::Client::builder()
            .pool_max_idle_per_host(0)
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(CLOSE_DEADLINE);
        if let Some(policy) = &self.egress {
            builder = builder.dns_resolver(crate::tools::web::ssrf::SsrfResolver::new(
                (**policy).clone(),
            ));
        }
        builder.build().map_err(|e| e.to_string())
    }
}

/// Every session the exit still has to end: those of live connections, and those left by
/// connections that went away before their `DELETE` was answered. Taken: each is sent once.
fn take_endings(registry: &Registry) -> Vec<Ending> {
    let live: Vec<Arc<HttpSession>> = std::mem::take(&mut *lock(&registry.open))
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .collect();
    let mut endings: Vec<Ending> = live.iter().filter_map(|s| s.ending()).collect();
    // Ended here; a later drop of these connections must not send them again.
    for session in &live {
        *lock(&session.open) = None;
    }
    endings.extend(
        std::mem::take(&mut *lock(&registry.orphans))
            .into_iter()
            .map(|(_, e)| e),
    );
    endings
}

/// Start ending every session still open (see the module doc); the returned waiter blocks until
/// they are all answered or `deadline` has passed since this call, whichever is first. Start it,
/// do the rest of the exit, then wait — so the HTTP round trips overlap the stdio sweep.
pub fn begin_close_all(deadline: Duration) -> impl FnOnce() {
    let endings = take_endings(&Registry::process());
    let until = Instant::now() + deadline;
    let (done, finished) = std::sync::mpsc::channel::<()>();
    if !endings.is_empty() {
        let spawned = std::thread::Builder::new()
            .name("mcp-http-exit".into())
            .spawn(move || {
                let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                else {
                    return;
                };
                runtime.block_on(async {
                    let sends = endings.iter().map(Ending::send);
                    let left = until.saturating_duration_since(Instant::now());
                    match tokio::time::timeout(left, futures::future::join_all(sends)).await {
                        Ok(results) => {
                            for (ending, result) in endings.iter().zip(results) {
                                if let Err(e) = result {
                                    tracing::debug!(uri = %ending.uri, error = %e, "ending an MCP session on exit failed");
                                }
                            }
                        }
                        Err(_) => tracing::debug!(
                            "ending MCP sessions on exit ran out of time; the servers will expire them"
                        ),
                    }
                });
                runtime.shutdown_timeout(Duration::from_millis(50));
                let _ = done.send(());
            });
        if spawned.is_err() {
            return Box::new(|| {}) as Box<dyn FnOnce()>;
        }
    } else {
        let _ = done.send(());
    }
    Box::new(move || {
        let _ = finished.recv_timeout(until.saturating_duration_since(Instant::now()));
    }) as Box<dyn FnOnce()>
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A loopback server that answers every request 200 and records each request's first line and
    /// `Mcp-Session-Id`.
    async fn server() -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = beyond_ai_test_support::ports::tokio_listener().await;
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let record = seen.clone();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let record = record.clone();
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
                    let mut buf = vec![0u8; 8192];
                    let n = stream.read(&mut buf).await.unwrap_or(0);
                    let head = String::from_utf8_lossy(&buf[..n]).to_ascii_lowercase();
                    let line = head.lines().next().unwrap_or_default().to_owned();
                    let session = head
                        .lines()
                        .find_map(|l| l.strip_prefix("mcp-session-id: "))
                        .unwrap_or_default();
                    record.lock().unwrap().push(format!("{line} {session}"));
                    let _ = stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        )
                        .await;
                });
            }
        });
        (url, seen)
    }

    /// What `registry` holds: live connections with a session open, and orphaned `DELETE`s.
    fn held(registry: &Registry) -> (usize, usize) {
        let live = lock(&registry.open)
            .iter()
            .filter(|s| s.strong_count() > 0)
            .count();
        (live, lock(&registry.orphans).len())
    }

    /// A connection that goes away with its session still open — its transport died before rmcp's
    /// own `DELETE` — is not kept by the process until it exits: the `DELETE` goes out at once, and
    /// once it is answered nothing of the connection is left.
    #[tokio::test]
    async fn a_connection_gone_with_its_session_open_ends_it_and_is_not_kept() {
        agent_core::ensure_provider();
        let registry = Arc::new(Registry::default());
        let (url, seen) = server().await;
        let session = HttpSession::new_in(registry.clone(), &url, None, None);
        session.opened("sess-gone", &HashMap::new());
        assert_eq!(held(&registry), (1, 0));
        drop(session);

        let deadline = Instant::now() + Duration::from_secs(10);
        while !seen.lock().unwrap().iter().any(|l| l.starts_with("delete")) {
            assert!(Instant::now() < deadline, "the DELETE never went out");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            seen.lock().unwrap().clone(),
            ["delete /mcp http/1.1 sess-gone"]
        );
        while held(&registry) != (0, 0) {
            assert!(Instant::now() < deadline, "{:?}", held(&registry));
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// A `DELETE` nobody answers is not kept either: once its attempt has timed out the entry goes,
    /// rather than living as long as the process.
    #[tokio::test]
    async fn an_unanswered_orphan_delete_is_dropped_after_its_timeout() {
        agent_core::ensure_provider();
        let registry = Arc::new(Registry {
            orphan_timeout: Duration::from_millis(300),
            ..Registry::default()
        });
        // A listener that accepts and never answers.
        let listener = beyond_ai_test_support::ports::tokio_listener().await;
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let _accepting = tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((stream, _)) = listener.accept().await {
                held.push(stream);
            }
        });
        let session = HttpSession::new_in(registry.clone(), &url, None, None);
        session.opened("sess-silent", &HashMap::new());
        drop(session);
        assert_eq!(held(&registry), (0, 1), "in flight");
        let deadline = Instant::now() + Duration::from_secs(10);
        while held(&registry) != (0, 0) {
            assert!(
                Instant::now() < deadline,
                "still kept: {:?}",
                held(&registry)
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// One rmcp ended itself is gone at once, and is not sent again on drop.
    #[tokio::test]
    async fn a_session_rmcp_ended_is_neither_kept_nor_sent_again() {
        let registry = Arc::new(Registry::default());
        let (url, seen) = server().await;
        let session = HttpSession::new_in(registry.clone(), &url, None, None);
        session.opened("sess-closed", &HashMap::new());
        session.closed("sess-closed");
        assert_eq!(held(&registry), (0, 0));
        drop(session);
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(seen.lock().unwrap().is_empty());
        assert_eq!(held(&registry), (0, 0));
    }
}
