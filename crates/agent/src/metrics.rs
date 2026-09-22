//! Prometheus metrics for a `serve --service` replica, on a listener of their own.
//!
//! # Why not `gateway/src/metrics.rs`
//!
//! That module registers on prometheus' **default** registry, because Pingora's built-in
//! `prometheus_http_service` exposes exactly that one. Its constructors also take gateway types
//! (`&crate::usage::Usage`, `crate::route::Provider`). Neither is reusable here, so this copies the
//! shape — `Arc<Metrics>`, children resolved once at boot — and shares no code. This module owns a
//! [`Registry`] of its own, so nothing a dependency happens to register on the default one can leak
//! onto a tenant-facing replica's scrape.
//!
//! # The label rule
//!
//! **No tenant, session, shard or workspace identifier appears in any label, ever.** That is two
//! problems at once: unbounded cardinality (a label per session would multiply every series by the
//! live-session count), and a tenancy leak (a scrape is not tenant-scoped, so one tenant's session
//! ids would be readable by whoever holds the scrape). Every label in this module is a **closed set**
//! fixed at compile time — a reason, a status class — and [`Metrics::new`] resolves each child once
//! so the hot paths never take a map lookup or a lock.
//!
//! # The listener
//!
//! [`serve_metrics`] answers `GET /metrics` and 404s everything else. It is deliberately *not* the
//! listener that serves the agent protocol: that one is reachable by tenants, and the scrape must
//! not be. [`parse_listen_addr`] refuses to bind anywhere but loopback for the same reason — a
//! metrics scraper belongs in the same task (an ECS `awsvpc` task shares a network namespace across
//! its containers, so a sidecar reaches `127.0.0.1` with no exposure beyond it).

use std::net::SocketAddr;
use std::sync::Arc;

use prometheus::{
    Encoder, Histogram, HistogramOpts, IntCounter, IntCounterVec, IntGauge, Opts, Registry,
    TextEncoder,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Longest request head this listener will read before giving up on a connection.
///
/// A scrape is a bare `GET /metrics HTTP/1.1` plus a handful of headers. Anything larger is not a
/// scraper, and reading it unbounded is how a listener with no body parsing still runs out of memory.
const MAX_REQUEST_BYTES: usize = 8 * 1024;

/// How long a connection has to send its request head. A scraper on loopback takes microseconds;
/// this exists so a connection that opens and says nothing cannot hold a task forever.
const READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Why a session's task ended — the closed label set of `agent_sessions_ended_total`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionEnd {
    /// The idle reaper reclaimed a detached session.
    IdleReap,
    /// Graceful shutdown ended it.
    Drain,
    /// The session's own loop finished (the client asked it to stop, or the connection closed and it
    /// ran out of work).
    Client,
    /// It ended on an error — including being fenced by a newer epoch.
    Error,
}

impl SessionEnd {
    fn as_str(self) -> &'static str {
        match self {
            Self::IdleReap => "idle_reap",
            Self::Drain => "drain",
            Self::Client => "client",
            Self::Error => "error",
        }
    }
}

/// Why taking a session's lock failed — the closed label set of `agent_lock_failures_total`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LockFailure {
    /// Another live owner holds it. The ordinary case behind a 503, not an error.
    Held,
    /// The attempt outlived `SESSION_LOCK_TIMEOUT` — usually a mount that stopped answering.
    Timeout,
    /// The filesystem refused the open or the lock.
    Io,
}

impl LockFailure {
    fn as_str(self) -> &'static str {
        match self {
            Self::Held => "held",
            Self::Timeout => "timeout",
            Self::Io => "io",
        }
    }
}

/// Why a connection was refused before it ever became a session — the closed label set of
/// `agent_refusals_total`.
///
/// Deliberately coarser than the HTTP status: `Misdirected` and `Unavailable` are both operational
/// signals an operator acts on differently (a routing bug versus a replica at capacity or a lock
/// still held), while the auth failures are one number because the distinction between them is a
/// tenant's problem, not the fleet's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// 401/403 — the grant did not verify, or named a tenant or session this connection may not have.
    Auth,
    /// 421 — this replica does not mount that session's shard.
    Misdirected,
    /// 503 — the lock is held elsewhere, or this replica is at `--max-live-sessions`.
    Unavailable,
    /// 400 — malformed request or grant.
    BadRequest,
}

impl Refusal {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auth => "auth",
            Self::Misdirected => "misdirected",
            Self::Unavailable => "unavailable",
            Self::BadRequest => "bad_request",
        }
    }
}

/// Every instrument a replica publishes, with its children resolved once.
///
/// Cloneable by `Arc` only — one of these exists per process, built in `main`.
pub struct Metrics {
    registry: Registry,

    /// Sessions this replica currently owns. The number `--max-live-sessions` is checked against,
    /// and the one that decides a failure's blast radius: everything counted here waits for a lock
    /// lease to expire if this replica dies.
    pub sessions_live: IntGauge,
    /// Prompts executing right now, across every session on this replica.
    ///
    /// The companion to `sessions_live`, and the one that predicts memory. A live session that is
    /// merely attached costs a few hundred KB; a session *running a turn* additionally holds its
    /// replay buffer, capped at `TURN_REPLAY_MAX_BYTES` (4 MiB). So this gauge — not the session
    /// count — is what multiplies into a replica's working set, and `--max-live-sessions` admits on
    /// the count rather than on this. An operator sizing a task, or explaining an OOM, needs both
    /// numbers and previously had only one.
    pub runs_in_flight: IntGauge,
    /// Sessions spawned since boot. With `sessions_ended_total` this gives churn, which is what
    /// separates "20,000 steady sessions" from "20,000 sessions a minute".
    pub sessions_spawned: IntCounter,
    sessions_ended: IntCounterVec,

    /// How long acquiring a session's lock took, successes only. The distribution that matters after
    /// a replica dies: a takeover cannot happen until the dead owner's lease lapses, so this is where
    /// that shows up as latency rather than as an error.
    pub lock_wait_seconds: Histogram,
    lock_failures: IntCounterVec,

    /// `session_superseded` frames sent: an owner discovering it has been fenced. Should be rare.
    ///
    /// The storage-side counters that would pair with it — appends, bytes, segment seals — are
    /// deliberately **absent** rather than defined and left at zero. They live below
    /// `session_store`'s `Log` seam, which has no handle on this, and a series that never moves is
    /// worse than a missing one: it reads as "this never happens" instead of "nobody measured".
    /// Wiring them means threading a handle through `RepoOptions`, which belongs in its own change.
    pub sessions_superseded: IntCounter,

    /// How long the `/readyz` shard probe took. The signal that a mount is degrading before it stops
    /// answering altogether.
    pub ready_probe_seconds: Histogram,

    /// OS threads in this process, sampled at scrape time rather than maintained.
    ///
    /// This is the number that says whether `/readyz` is leaking. A readiness probe that does a
    /// blocking `stat` per request against a mount whose target has gone parks one blocking-pool
    /// thread per probe and never gets it back: the replica climbs toward the pool ceiling and then
    /// stops answering anything at all, health checks included. Single-flighting the probe keeps
    /// this flat while the mount hangs, and *flat* is the claim — which nothing outside the task can
    /// verify, because `/proc` is not shared across a container boundary and a thread count is not
    /// visible from a socket. Sampled here so a sidecar can see it.
    ///
    /// Linux only; elsewhere it reads 0, which is why it is a sampled gauge and not a counter.
    threads: IntGauge,

    refusals: IntCounterVec,
}

impl Metrics {
    /// Build the registry and resolve every child. Fails only if two instruments collide on a name,
    /// which is a programming error caught the first time this runs.
    pub fn new() -> Result<Arc<Self>, prometheus::Error> {
        let registry = Registry::new();

        let sessions_live = IntGauge::with_opts(Opts::new(
            "agent_sessions_live",
            "Sessions this replica currently owns.",
        ))?;
        let runs_in_flight = IntGauge::with_opts(Opts::new(
            "agent_runs_in_flight",
            "Prompts executing right now across every session on this replica.",
        ))?;
        let sessions_spawned = IntCounter::with_opts(Opts::new(
            "agent_sessions_spawned_total",
            "Sessions spawned since boot.",
        ))?;
        let sessions_ended = IntCounterVec::new(
            Opts::new(
                "agent_sessions_ended_total",
                "Session tasks that exited, by reason.",
            ),
            &["reason"],
        )?;
        let lock_wait_seconds = Histogram::with_opts(HistogramOpts::new(
            "agent_lock_wait_seconds",
            "Time to acquire a session's advisory lock, successes only.",
        ))?;
        let lock_failures = IntCounterVec::new(
            Opts::new(
                "agent_lock_failures_total",
                "Session lock acquisitions that did not succeed, by reason.",
            ),
            &["reason"],
        )?;
        let sessions_superseded = IntCounter::with_opts(Opts::new(
            "agent_sessions_superseded_total",
            "Sessions that discovered they had been fenced by a newer epoch.",
        ))?;
        let ready_probe_seconds = Histogram::with_opts(HistogramOpts::new(
            "agent_ready_probe_seconds",
            "Duration of the /readyz shard probe.",
        ))?;
        let threads = IntGauge::with_opts(Opts::new(
            "agent_threads",
            "OS threads in this process, sampled at scrape time.",
        ))?;
        let refusals = IntCounterVec::new(
            Opts::new(
                "agent_refusals_total",
                "Connections refused before becoming a session, by reason.",
            ),
            &["reason"],
        )?;

        registry.register(Box::new(sessions_live.clone()))?;
        registry.register(Box::new(runs_in_flight.clone()))?;
        registry.register(Box::new(sessions_spawned.clone()))?;
        registry.register(Box::new(sessions_ended.clone()))?;
        registry.register(Box::new(lock_wait_seconds.clone()))?;
        registry.register(Box::new(lock_failures.clone()))?;
        registry.register(Box::new(sessions_superseded.clone()))?;
        registry.register(Box::new(ready_probe_seconds.clone()))?;
        registry.register(Box::new(threads.clone()))?;
        registry.register(Box::new(refusals.clone()))?;

        // Touch every child once so a series exists at zero rather than appearing on first use. An
        // alert on `rate(agent_refusals_total{reason="unavailable"}[5m])` should read zero on a
        // healthy replica, not "no data" — those mean different things to whoever is paged.
        for r in [
            SessionEnd::IdleReap,
            SessionEnd::Drain,
            SessionEnd::Client,
            SessionEnd::Error,
        ] {
            sessions_ended.with_label_values(&[r.as_str()]);
        }
        for r in [LockFailure::Held, LockFailure::Timeout, LockFailure::Io] {
            lock_failures.with_label_values(&[r.as_str()]);
        }
        for r in [
            Refusal::Auth,
            Refusal::Misdirected,
            Refusal::Unavailable,
            Refusal::BadRequest,
        ] {
            refusals.with_label_values(&[r.as_str()]);
        }

        Ok(Arc::new(Self {
            registry,
            sessions_live,
            runs_in_flight,
            sessions_spawned,
            sessions_ended,
            lock_wait_seconds,
            lock_failures,
            sessions_superseded,
            ready_probe_seconds,
            threads,
            refusals,
        }))
    }

    pub fn session_ended(&self, reason: SessionEnd) {
        self.sessions_ended
            .with_label_values(&[reason.as_str()])
            .inc();
    }

    pub fn lock_failed(&self, reason: LockFailure) {
        self.lock_failures
            .with_label_values(&[reason.as_str()])
            .inc();
    }

    pub fn refused(&self, reason: Refusal) {
        self.refusals.with_label_values(&[reason.as_str()]).inc();
    }

    /// The scrape body, in the Prometheus text exposition format.
    pub fn encode(&self) -> Vec<u8> {
        // Sampled here, not tracked: threads are created and retired by the runtime, which has no
        // hook to count them through, and a scrape is the only moment the number is wanted.
        self.threads.set(os_threads());
        let mut buf = Vec::new();
        let encoder = TextEncoder::new();
        // Encoding fails only on a broken writer; a `Vec` is not one.
        let _ = encoder.encode(&self.registry.gather(), &mut buf);
        buf
    }
}

/// OS threads in this process, or 0 where the platform does not say.
///
/// `/proc/self/status` rather than counting `/proc/self/task`: one read and one parse, against one
/// `readdir` plus an allocation per thread — and this runs on a scrape, which a replica serving
/// tenants should not be paying a per-thread cost for.
fn os_threads() -> i64 {
    #[cfg(target_os = "linux")]
    {
        let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
            return 0;
        };
        status
            .lines()
            .find_map(|l| l.strip_prefix("Threads:"))
            .and_then(|n| n.trim().parse().ok())
            .unwrap_or(0)
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

/// Parse `--metrics-listen`, refusing anything but a loopback address.
///
/// The scrape carries operational detail about every tenant on the replica, and the replica is
/// reachable by tenants. Binding it to a routable interface would publish that; a scraper that needs
/// it runs beside the replica (in an ECS `awsvpc` task, containers share the network namespace, so a
/// sidecar reaches `127.0.0.1` without the port existing anywhere else). Refused rather than warned,
/// because a warning in a log is not a thing anyone reads before the port is already open.
pub fn parse_listen_addr(s: &str) -> Result<SocketAddr, String> {
    let addr: SocketAddr = s.parse().map_err(|_| {
        format!("--metrics-listen {s:?}: expected <ip>:<port>, e.g. 127.0.0.1:9095")
    })?;
    if !addr.ip().is_loopback() {
        return Err(format!(
            "--metrics-listen {s:?}: must be a loopback address (127.0.0.1 or [::1]) — the scrape \
             describes every tenant on this replica and the replica is reachable by tenants. Run \
             the scraper beside it."
        ));
    }
    Ok(addr)
}

/// Serve `GET /metrics` on `listener` until the process ends.
///
/// One connection at a time is deliberate: this is a scrape endpoint on loopback with one client,
/// and a connection per task would let an errant scraper spawn tasks faster than they retire. Every
/// error closes the connection and continues — a metrics listener must never be able to end the
/// process it is reporting on.
pub async fn serve_metrics(listener: TcpListener, metrics: Arc<Metrics>) {
    loop {
        let Ok((mut stream, _)) = listener.accept().await else {
            continue;
        };
        let mut buf = Vec::new();
        let mut chunk = [0u8; 1024];
        let head = loop {
            match tokio::time::timeout(READ_TIMEOUT, stream.read(&mut chunk)).await {
                Ok(Ok(0)) | Err(_) | Ok(Err(_)) => break None,
                Ok(Ok(n)) => {
                    buf.extend_from_slice(&chunk[..n]);
                    if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                        break Some(buf);
                    }
                    if buf.len() > MAX_REQUEST_BYTES {
                        break None;
                    }
                }
            }
        };
        let Some(head) = head else {
            continue;
        };
        let line = head
            .split(|b| *b == b'\n')
            .next()
            .map(|l| String::from_utf8_lossy(l).trim().to_owned())
            .unwrap_or_default();
        let mut parts = line.split(' ');
        let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));

        let response = if method == "GET" && path == "/metrics" {
            let body = metrics.encode();
            let mut out = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: \
                 {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .into_bytes();
            out.extend_from_slice(&body);
            out
        } else {
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
        };
        let _ = stream.write_all(&response).await;
        let _ = stream.flush().await;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn loopback_is_accepted_and_anything_else_is_refused() {
        assert!(parse_listen_addr("127.0.0.1:9095").is_ok());
        assert!(parse_listen_addr("[::1]:9095").is_ok());
        let err = parse_listen_addr("0.0.0.0:9095").unwrap_err();
        assert!(err.contains("loopback"), "got: {err}");
        let err = parse_listen_addr("10.0.0.4:9095").unwrap_err();
        assert!(err.contains("loopback"), "got: {err}");
        assert!(parse_listen_addr("not-an-addr").is_err());
    }

    #[test]
    fn no_label_can_carry_a_tenant_or_session_identifier() {
        // The guard for the rule in this module's doc: every label name is a closed set fixed here.
        // A future instrument that adds `tenant` or `session_id` fails this test rather than shipping
        // a scrape that leaks one tenant's ids to whoever holds it.
        let m = Metrics::new().unwrap();
        m.session_ended(SessionEnd::Drain);
        m.lock_failed(LockFailure::Held);
        m.refused(Refusal::Unavailable);
        let text = String::from_utf8(m.encode()).unwrap();
        for forbidden in ["tenant", "session_id", "shard", "workspace"] {
            assert!(
                !text.contains(&format!("{forbidden}=")),
                "metrics exposed a {forbidden} label:\n{text}"
            );
        }
    }

    #[test]
    fn every_child_exists_at_zero_before_first_use() {
        // "No data" and "zero" page differently. Each closed-set child is touched at construction.
        let text = String::from_utf8(Metrics::new().unwrap().encode()).unwrap();
        for series in [
            r#"agent_sessions_ended_total{reason="drain"} 0"#,
            r#"agent_lock_failures_total{reason="timeout"} 0"#,
            r#"agent_refusals_total{reason="misdirected"} 0"#,
        ] {
            assert!(text.contains(series), "missing {series} in:\n{text}");
        }
    }

    /// Drive the real listener over a real socket: the encoder, the content type a scraper insists
    /// on, and the 404 that keeps this endpoint from answering anything else.
    async fn request(addr: SocketAddr, path: &str) -> String {
        use tokio::io::AsyncWriteExt as _;
        let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
        s.write_all(format!("GET {path} HTTP/1.1\r\nHost: x\r\n\r\n").as_bytes())
            .await
            .unwrap();
        let mut out = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut s, &mut out)
            .await
            .unwrap();
        String::from_utf8_lossy(&out).into_owned()
    }

    #[tokio::test]
    async fn the_listener_serves_metrics_and_refuses_every_other_path() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let m = Metrics::new().unwrap();
        m.sessions_live.set(7);
        tokio::spawn(serve_metrics(listener, Arc::clone(&m)));

        let scrape = request(addr, "/metrics").await;
        assert!(scrape.starts_with("HTTP/1.1 200 OK"), "{scrape}");
        assert!(
            scrape.contains("Content-Type: text/plain; version=0.0.4"),
            "a scraper needs the exposition content type: {scrape}"
        );
        assert!(scrape.contains("agent_sessions_live 7"), "{scrape}");

        // `/livez` and `/readyz` belong to the tenant-facing listener. This one knows one path.
        let other = request(addr, "/livez").await;
        assert!(other.starts_with("HTTP/1.1 404"), "{other}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_thread_gauge_is_sampled_on_every_scrape() {
        let m = Metrics::new().unwrap();
        // Zero before anything encodes: the gauge is sampled, never maintained, so an un-scraped
        // registry has no opinion about it.
        assert_eq!(m.threads.get(), 0);

        let before = String::from_utf8(m.encode()).unwrap();
        let sampled = m.threads.get();
        assert!(
            sampled > 0,
            "a running process has at least one thread: {before}"
        );
        assert!(
            before.contains(&format!("agent_threads {sampled}")),
            "{before}"
        );

        // And every scrape re-samples rather than latching the first reading. Asserted by
        // poisoning the gauge instead of by spawning a thread and watching the number rise: this
        // test shares a process with the rest of the suite, whose own threads come and go, so a
        // strict inequality here would be grading the test runner. What C8 needs from this gauge is
        // that a *later* scrape tells the truth about *now*, and that is exactly what this checks.
        m.threads.set(999_999);
        let after = String::from_utf8(m.encode()).unwrap();
        assert!(
            m.threads.get() > 0 && m.threads.get() < 999_999,
            "a scrape must re-sample, not report what was there before: {after}"
        );
    }

    #[test]
    fn the_in_flight_gauge_follows_transitions_not_assertions() {
        // The retry path re-asserts a run's `running` flag while the run is already in flight. A
        // gauge that counted assertions would double it and never come back down — which is worse
        // than no gauge, because the number an operator sizes a task against would drift upward with
        // every retry and never recover.
        let m = Metrics::new().unwrap();
        let running = std::sync::atomic::AtomicBool::new(false);
        let bump = |m: &Metrics| {
            if !running.swap(true, std::sync::atomic::Ordering::Relaxed) {
                m.runs_in_flight.inc();
            }
        };
        bump(&m);
        bump(&m); // the retry's re-assertion
        bump(&m);
        assert_eq!(m.runs_in_flight.get(), 1, "one prompt is one run in flight");

        if running.swap(false, std::sync::atomic::Ordering::Relaxed) {
            m.runs_in_flight.dec();
        }
        assert_eq!(m.runs_in_flight.get(), 0, "and it comes back down");
    }

    #[test]
    fn counters_and_gauges_reach_the_scrape() {
        let m = Metrics::new().unwrap();
        m.sessions_live.set(3);
        m.sessions_spawned.inc();
        m.sessions_superseded.inc();
        m.refused(Refusal::Auth);
        m.refused(Refusal::Auth);
        let text = String::from_utf8(m.encode()).unwrap();
        assert!(text.contains("agent_sessions_live 3"), "{text}");
        assert!(text.contains("agent_sessions_spawned_total 1"), "{text}");
        assert!(text.contains("agent_sessions_superseded_total 1"), "{text}");
        assert!(
            text.contains(r#"agent_refusals_total{reason="auth"} 2"#),
            "{text}"
        );
    }
}
