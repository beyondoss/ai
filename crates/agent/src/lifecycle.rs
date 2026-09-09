//! Run lifecycle — a first-class, host-emitted record of what a run did.
//!
//! A run's plan, current tool, closing prose and outcome are otherwise only visible on a session
//! connection that a consumer has to hold for the whole duration. This module makes that record a
//! structured output of the host, independent of any client.
//!
//! **This is telemetry, not a tool.** The runtime emits it. The model is never asked and cannot
//! influence it. Facts come from the loop (tool start/end, step counter, todo list, last visible
//! assistant text) and from the host (retry, persist, the response the client was told).
//!
//! Unconfigured: [`NoLifecycle`]. Zero I/O, zero behavioural difference. A slow or unreachable
//! consumer is logged and dropped; it never adds latency to the turn.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use agent_core::AgentEvent;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{Notify, oneshot};

use crate::exec_endpoint::HttpExecRunner;

/// Visible closing-summary budget, in Unicode scalars. Truncated on a scalar boundary.
pub const SUMMARY_MAX_CHARS: usize = 4000;
/// Drop `structured_output` from the terminal event rather than POST a huge payload.
const STRUCTURED_OUTPUT_MAX_BYTES: usize = 8192;
/// Durable started/terminal queue. Progress lives in a separate latest-wins slot and cannot fill this.
const DURABLE_CAP: usize = 256;
/// How long a CLI/`drain` waits for in-flight POSTs before giving up.
const DRAIN_TIMEOUT: Duration = Duration::from_millis(500);
/// Per-request HTTP timeout. Progress is not retried; started/terminal retry a few times inside this.
const HTTP_TIMEOUT: Duration = Duration::from_secs(2);
const HTTP_RETRIES: u32 = 3;
/// Default in-flight catch-up interval. Tool start/end, todos, and turn-end emit a progress sample
/// immediately; this timer only re-POSTs if the sample has changed since the last POST. A long
/// `bash` is one progress event, not one per tick. Silence means unchanged, not dead.
///
/// The wire event is `progress`. **Heartbeat** names this optional timer, not the event. `0` on
/// `--lifecycle-heartbeat-secs` disables the timer without touching sample-on-change.
pub const HEARTBEAT: Duration = Duration::from_secs(10);

/// Timer for the in-flight catch-up, or `None` when disabled (`Duration::ZERO`). The caller must
/// consume the first tick so the first sample is one interval after `started`, not stacked on it.
pub fn heartbeat_interval(every: Duration) -> Option<tokio::time::Interval> {
    if every.is_zero() {
        return None;
    }
    let mut interval = tokio::time::interval(every);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    Some(interval)
}

/// The host's seam for run lifecycle. Follows the crate idiom: a trait with a no-op default.
///
/// Lives here, not in `agent-core`. The outcome of a run depends on the retry layer, on whether the
/// transcript persisted, and on what response the client received — none of which the loop can see.
#[async_trait]
pub trait RunLifecycle: Send + Sync {
    /// `false` for [`NoLifecycle`]: hosts skip minting a [`Run`] entirely, so an unconfigured process
    /// does no extra work on the turn path.
    fn enabled(&self) -> bool {
        true
    }

    /// Non-blocking. The implementation must not await the network on this call.
    /// [`NoLifecycle`]'s default is a no-op, so an unconfigured host can share this trait object
    /// without a separate code path at every emit site.
    fn emit(&self, _event: RunEvent) {}

    /// Wait (bounded) for outstanding durable events to leave the process. No-op by default.
    async fn drain(&self) {}
}

/// Inert default: zero I/O.
pub struct NoLifecycle;

#[async_trait]
impl RunLifecycle for NoLifecycle {
    fn enabled(&self) -> bool {
        false
    }
}

/// Construct the emitter. `None` URL → [`NoLifecycle`]. Validates the URL and every header here, so a
/// typo fails at startup rather than at the end of the first run.
pub fn open(url: Option<&str>, headers: &[String]) -> Result<Arc<dyn RunLifecycle>, String> {
    let Some(url) = url.filter(|u| !u.is_empty()) else {
        return Ok(Arc::new(NoLifecycle));
    };
    let mut parsed = Vec::with_capacity(headers.len());
    for raw in headers {
        parsed.push(HttpExecRunner::parse_header(raw)?);
    }
    Ok(Arc::new(HttpLifecycle::spawn(url, parsed)?))
}

/// Where the worker POSTs. `http`/`https` are a network peer; `unix:` is a socket on this machine
/// (no host to name, no TLS to terminate). Reqwest still wants an `http://` request URL when bound
/// to a Unix socket, so `unix:` is rewritten to `http://localhost/` at this boundary.
#[derive(Debug)]
struct Endpoint {
    request_url: String,
    unix_socket: Option<PathBuf>,
}

fn parse_endpoint(url: &str) -> Result<Endpoint, String> {
    let parsed =
        reqwest::Url::parse(url).map_err(|e| format!("invalid lifecycle URL {url:?}: {e}"))?;
    match parsed.scheme() {
        "http" | "https" => Ok(Endpoint {
            request_url: url.to_string(),
            unix_socket: None,
        }),
        "unix" => parse_unix(&parsed, url),
        other => Err(format!(
            "lifecycle URL must be http, https, or unix, got {other:?}"
        )),
    }
}

fn parse_unix(parsed: &reqwest::Url, raw: &str) -> Result<Endpoint, String> {
    #[cfg(not(unix))]
    {
        let _ = parsed;
        return Err(format!(
            "lifecycle unix: URLs are only supported on unix, got {raw:?}"
        ));
    }
    #[cfg(unix)]
    {
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err(format!(
                "lifecycle unix: URL must not carry userinfo, got {raw:?}"
            ));
        }
        if let Some(host) = parsed.host_str()
            && host != "localhost"
        {
            return Err(format!(
                "lifecycle unix: URL names a host ({host:?}); use http(s):// for a network peer, or unix:///path for a local socket"
            ));
        }
        if parsed.query().is_some() || parsed.fragment().is_some() {
            return Err(format!(
                "lifecycle unix: URL must not have a query or fragment, got {raw:?}"
            ));
        }
        let path = parsed.path();
        if path.is_empty() || path == "/" {
            return Err(format!(
                "lifecycle unix: URL must name a socket path, got {raw:?}"
            ));
        }
        if !path.starts_with('/') {
            return Err(format!(
                "lifecycle unix: socket path must be absolute, got {path:?}"
            ));
        }
        Ok(Endpoint {
            // Dummy origin: the client is bound to the socket and never dials this host.
            request_url: "http://localhost/".to_string(),
            unix_socket: Some(PathBuf::from(path)),
        })
    }
}

/// One lifecycle event. Tagged `type` on the wire so a consumer can switch without a wrapping envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RunEvent {
    Started {
        run_id: String,
        session_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        command_id: Option<String>,
        ts: u64,
        model: String,
    },
    Progress {
        run_id: String,
        session_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        command_id: Option<String>,
        ts: u64,
        steps: u32,
        /// Currently executing tool **names** (a batch can run more than one). Never arguments.
        tools: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        todos: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        depth: Option<u32>,
    },
    Succeeded {
        run_id: String,
        session_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        command_id: Option<String>,
        ts: u64,
        steps: u32,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        refused: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        summary: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        structured_output: Option<Value>,
    },
    Failed {
        run_id: String,
        session_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        command_id: Option<String>,
        ts: u64,
        steps: u32,
        error: String,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        refused: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        summary: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        structured_output: Option<Value>,
    },
    Aborted {
        run_id: String,
        session_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        command_id: Option<String>,
        ts: u64,
        steps: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        summary: Option<String>,
    },
}

impl RunEvent {
    pub fn run_id(&self) -> &str {
        match self {
            Self::Started { run_id, .. }
            | Self::Progress { run_id, .. }
            | Self::Succeeded { run_id, .. }
            | Self::Failed { run_id, .. }
            | Self::Aborted { run_id, .. } => run_id,
        }
    }

    fn is_progress(&self) -> bool {
        matches!(self, Self::Progress { .. })
    }

    fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Succeeded { .. } | Self::Failed { .. } | Self::Aborted { .. }
        )
    }
}

/// One user-facing run: a `serve` prompt (including whole-run retries) or one CLI `run` invocation.
///
/// Structural properties live here, not at a call site: exactly one `started` and one terminal,
/// progress after terminal dropped, a second terminal a no-op.
pub struct Run {
    sink: Arc<dyn RunLifecycle>,
    run_id: String,
    session_id: String,
    command_id: Option<String>,
    model: String,
    /// 0 = idle, 1 = in flight, 2 = terminal emitted.
    state: std::sync::atomic::AtomicU8,
    sample: Mutex<ProgressSample>,
    /// Last sample that was actually emitted. Seeded empty so a timer tick with nothing in flight
    /// does not POST a no-op `progress` on top of `started`.
    last_emitted: Mutex<Option<ProgressSample>>,
}

#[derive(Clone, Default, PartialEq)]
struct ProgressSample {
    steps: u32,
    /// `(tool_use id, name)` in start order, so two concurrent `grep`s both appear.
    tools: Vec<(String, String)>,
    todos: Option<Value>,
    attempt: u32,
    depth: u32,
}

impl Run {
    /// Mint a `run_id`, emit `started`. No-op-safe: callers still check [`RunLifecycle::enabled`]
    /// so an unconfigured host never constructs this.
    pub fn begin(
        sink: Arc<dyn RunLifecycle>,
        session_id: impl Into<String>,
        command_id: Option<String>,
        model: impl Into<String>,
    ) -> Arc<Self> {
        let this = Arc::new(Self {
            sink,
            run_id: crate::session_store::new_id(),
            session_id: session_id.into(),
            command_id,
            model: model.into(),
            state: std::sync::atomic::AtomicU8::new(0),
            sample: Mutex::new(ProgressSample::default()),
            last_emitted: Mutex::new(Some(ProgressSample::default())),
        });
        this.started();
        this
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    fn started(&self) {
        if self
            .state
            .compare_exchange(
                0,
                1,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_err()
        {
            return;
        }
        self.sink.emit(RunEvent::Started {
            run_id: self.run_id.clone(),
            session_id: self.session_id.clone(),
            command_id: self.command_id.clone(),
            ts: now_ts(),
            model: self.model.clone(),
        });
    }

    /// Fold a loop event into the sample. Tool edges, todos, and turn-end emit immediately.
    pub fn observe(&self, ev: &AgentEvent) {
        match ev {
            AgentEvent::ToolStart { id, name, .. } => {
                {
                    let mut s = lock(&self.sample);
                    s.tools.push((id.clone(), name.clone()));
                }
                self.emit_progress();
            }
            AgentEvent::ToolEnd { id, .. } => {
                {
                    let mut s = lock(&self.sample);
                    s.tools.retain(|(i, _)| i != id);
                }
                self.emit_progress();
            }
            AgentEvent::ToolProgress {
                name,
                details: Some(d),
                ..
            } if name == crate::tools::todo::NAME => {
                if let Some(todos) = d.get("todos") {
                    lock(&self.sample).todos = Some(todos.clone());
                    self.emit_progress();
                }
            }
            AgentEvent::TurnEnd { step, .. } => {
                lock(&self.sample).steps = *step;
                self.emit_progress();
            }
            _ => {}
        }
    }

    pub fn set_attempt(&self, attempt: u32) {
        lock(&self.sample).attempt = attempt;
    }

    pub fn set_steps(&self, steps: u32) {
        lock(&self.sample).steps = steps;
    }

    /// Timer path: re-POST the latest sample only if it changed. Hosts call this on the interval.
    pub fn heartbeat(&self) {
        self.emit_progress();
    }

    /// Latest-wins progress sample. Dropped if the run is not in flight, or if this sample was
    /// already the last one posted (an unchanged timer tick is not a new event).
    fn emit_progress(&self) {
        if self.state.load(std::sync::atomic::Ordering::Acquire) != 1 {
            return;
        }
        let s = lock(&self.sample).clone();
        {
            let mut last = lock(&self.last_emitted);
            if last.as_ref() == Some(&s) {
                return;
            }
            *last = Some(s.clone());
        }
        self.sink.emit(RunEvent::Progress {
            run_id: self.run_id.clone(),
            session_id: self.session_id.clone(),
            command_id: self.command_id.clone(),
            ts: now_ts(),
            steps: s.steps,
            tools: s.tools.iter().map(|(_, n)| n.clone()).collect(),
            attempt: (s.attempt > 0).then_some(s.attempt),
            todos: s.todos,
            depth: (s.depth > 0).then_some(s.depth),
        });
    }

    pub fn succeeded(
        &self,
        steps: u32,
        refused: bool,
        summary: Option<String>,
        structured_output: Option<Value>,
    ) {
        self.terminal(RunEvent::Succeeded {
            run_id: self.run_id.clone(),
            session_id: self.session_id.clone(),
            command_id: self.command_id.clone(),
            ts: now_ts(),
            steps,
            refused,
            summary: cap_summary(summary),
            structured_output: cap_structured(structured_output),
        });
    }

    pub fn failed(
        &self,
        steps: u32,
        error: impl Into<String>,
        refused: bool,
        summary: Option<String>,
        structured_output: Option<Value>,
    ) {
        self.terminal(RunEvent::Failed {
            run_id: self.run_id.clone(),
            session_id: self.session_id.clone(),
            command_id: self.command_id.clone(),
            ts: now_ts(),
            steps,
            error: error.into(),
            refused,
            summary: cap_summary(summary),
            structured_output: cap_structured(structured_output),
        });
    }

    pub fn aborted(&self, steps: u32, summary: Option<String>) {
        self.terminal(RunEvent::Aborted {
            run_id: self.run_id.clone(),
            session_id: self.session_id.clone(),
            command_id: self.command_id.clone(),
            ts: now_ts(),
            steps,
            summary: cap_summary(summary),
        });
    }

    fn terminal(&self, event: RunEvent) {
        if self
            .state
            .compare_exchange(
                1,
                2,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_err()
        {
            return;
        }
        self.sink.emit(event);
    }

    pub async fn drain(&self) {
        self.sink.drain().await;
    }
}

/// Closing prose: visible `Text` blocks only, capped. Thinking / redacted-thinking never reach here
/// because [`crate::agents::last_assistant_text`] already filters them out.
pub fn closing_summary(session: &agent_core::Session) -> Option<String> {
    let text = crate::agents::last_assistant_text(session);
    cap_summary((!text.is_empty()).then_some(text))
}

/// Truncate `s` to `max` Unicode scalars, cutting on a scalar boundary.
pub fn truncate_chars(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        None => s.to_string(),
        Some((idx, _)) => s[..idx].to_string(),
    }
}

fn cap_summary(summary: Option<String>) -> Option<String> {
    summary
        .map(|s| truncate_chars(&s, SUMMARY_MAX_CHARS))
        .filter(|s| !s.is_empty())
}

fn cap_structured(value: Option<Value>) -> Option<Value> {
    let v = value?;
    match serde_json::to_vec(&v) {
        Ok(bytes) if bytes.len() <= STRUCTURED_OUTPUT_MAX_BYTES => Some(v),
        Ok(_) => {
            tracing::warn!("lifecycle: dropping oversized structured_output");
            None
        }
        Err(_) => None,
    }
}

fn now_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// HTTP emitter: one outbound worker, HTTP as the wire.
///
/// Started/terminal share a small durable queue (retry, then log+drop). Progress is one slot per
/// `run_id` (overwrite, no retry). The host never awaits the network.
struct HttpLifecycle {
    inner: Arc<HttpInner>,
}

struct HttpInner {
    url: String,
    headers: Vec<(String, String)>,
    client: reqwest::Client,
    durable: Mutex<VecDeque<RunEvent>>,
    progress: Mutex<HashMap<String, RunEvent>>,
    /// Run ids whose terminal is queued or in flight. Drops the TOCTOU between [`Run`]'s latch
    /// (which already refuses progress after terminal) and this `emit`. Evicted once that terminal
    /// has been POSTed or dropped — not retained for process lifetime.
    terminated: Mutex<HashSet<String>>,
    notify: Notify,
    drain: Mutex<Vec<oneshot::Sender<()>>>,
}

impl HttpLifecycle {
    fn spawn(url: &str, headers: Vec<(String, String)>) -> Result<Self, String> {
        let endpoint = parse_endpoint(url)?;
        agent_core::ensure_provider();
        let mut builder = reqwest::Client::builder().timeout(HTTP_TIMEOUT);
        #[cfg(unix)]
        if let Some(ref sock) = endpoint.unix_socket {
            builder = builder.unix_socket(sock.clone());
        }
        let client = builder
            .build()
            .map_err(|e| format!("lifecycle HTTP client: {e}"))?;
        let inner = Arc::new(HttpInner {
            url: endpoint.request_url,
            headers,
            client,
            durable: Mutex::new(VecDeque::new()),
            progress: Mutex::new(HashMap::new()),
            terminated: Mutex::new(HashSet::new()),
            notify: Notify::new(),
            drain: Mutex::new(Vec::new()),
        });
        let worker = Arc::clone(&inner);
        tokio::spawn(async move { worker_loop(worker).await });
        Ok(Self { inner })
    }
}

#[async_trait]
impl RunLifecycle for HttpLifecycle {
    fn emit(&self, event: RunEvent) {
        if event.is_progress() {
            let id = event.run_id().to_string();
            if lock(&self.inner.terminated).contains(&id) {
                return;
            }
            lock(&self.inner.progress).insert(id, event);
            self.inner.notify.notify_one();
            return;
        }
        if event.is_terminal() {
            self.inner.remember_terminal(event.run_id());
        }
        {
            let mut q = lock(&self.inner.durable);
            if q.len() >= DURABLE_CAP {
                tracing::warn!(
                    run_id = event.run_id(),
                    "lifecycle durable queue full; dropping event"
                );
                if event.is_terminal() {
                    self.inner.forget_terminal(event.run_id());
                }
                return;
            }
            q.push_back(event);
        }
        self.inner.notify.notify_one();
    }

    async fn drain(&self) {
        let (tx, rx) = oneshot::channel();
        lock(&self.inner.drain).push(tx);
        self.inner.notify.notify_one();
        let _ = tokio::time::timeout(DRAIN_TIMEOUT, rx).await;
    }
}

impl HttpInner {
    fn remember_terminal(&self, run_id: &str) {
        lock(&self.terminated).insert(run_id.to_string());
        lock(&self.progress).remove(run_id);
    }

    fn forget_terminal(&self, run_id: &str) {
        lock(&self.terminated).remove(run_id);
    }
}

async fn worker_loop(inner: Arc<HttpInner>) {
    loop {
        inner.notify.notified().await;
        loop {
            let batch: Vec<RunEvent> = lock(&inner.durable).drain(..).collect();
            let mut flushed_terminal: Vec<String> = Vec::new();
            for ev in &batch {
                post_durable(&inner, ev).await;
                if ev.is_terminal() {
                    flushed_terminal.push(ev.run_id().to_string());
                }
            }
            // Filter late progress *before* forgetting the id, so a sample that raced the latch
            // into the slot is dropped rather than POSTed after the terminal.
            let samples: Vec<RunEvent> = {
                let terminated = lock(&inner.terminated);
                lock(&inner.progress)
                    .drain()
                    .filter(|(id, _)| !terminated.contains(id))
                    .map(|(_, ev)| ev)
                    .collect()
            };
            for id in flushed_terminal {
                inner.forget_terminal(&id);
            }
            for ev in samples {
                let _ = post_once(&inner, &ev).await;
            }
            let idle = lock(&inner.durable).is_empty() && lock(&inner.progress).is_empty();
            if idle {
                for tx in lock(&inner.drain).drain(..) {
                    let _ = tx.send(());
                }
                break;
            }
        }
    }
}

async fn post_durable(inner: &HttpInner, event: &RunEvent) {
    let mut delay = Duration::from_millis(50);
    for attempt in 1..=HTTP_RETRIES {
        match post_once(inner, event).await {
            Ok(()) => return,
            Err(e) => {
                if attempt == HTTP_RETRIES {
                    tracing::warn!(
                        run_id = event.run_id(),
                        error = %e,
                        "lifecycle durable event dropped after retries"
                    );
                    return;
                }
                tokio::time::sleep(delay).await;
                delay = delay.saturating_mul(2);
            }
        }
    }
}

async fn post_once(inner: &HttpInner, event: &RunEvent) -> Result<(), String> {
    let mut req = inner.client.post(&inner.url).json(event);
    for (name, value) in &inner.headers {
        req = req.header(name.as_str(), value.as_str());
    }
    let resp = req.send().await.map_err(|e| e.to_string())?;
    if resp.status().is_success() {
        Ok(())
    } else {
        Err(format!("HTTP {}", resp.status()))
    }
}

#[cfg(test)]
pub(crate) struct RecordingLifecycle {
    pub events: Mutex<Vec<RunEvent>>,
}

#[cfg(test)]
impl RecordingLifecycle {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            events: Mutex::new(Vec::new()),
        })
    }

    pub fn snapshot(&self) -> Vec<RunEvent> {
        lock(&self.events).clone()
    }
}

#[cfg(test)]
#[async_trait]
impl RunLifecycle for RecordingLifecycle {
    fn emit(&self, event: RunEvent) {
        lock(&self.events).push(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::{ContentBlock, Message, Session};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

    fn rec_run() -> (Arc<RecordingLifecycle>, Arc<Run>) {
        let rec = RecordingLifecycle::new();
        let run = Run::begin(rec.clone(), "sess-1", Some("cmd-1".into()), "claude-test");
        (rec, run)
    }

    #[test]
    fn one_started_and_one_terminal() {
        let (rec, run) = rec_run();
        run.succeeded(3, false, Some("done".into()), None);
        run.succeeded(9, false, Some("again".into()), None);
        run.aborted(1, None);
        let events = rec.snapshot();
        assert_eq!(events.len(), 2, "{events:#?}");
        assert!(matches!(events[0], RunEvent::Started { .. }));
        assert!(matches!(
            events[1],
            RunEvent::Succeeded {
                steps: 3,
                refused: false,
                ..
            }
        ));
        assert_eq!(events[0].run_id(), events[1].run_id());
    }

    #[test]
    fn progress_after_terminal_is_dropped() {
        let (rec, run) = rec_run();
        run.aborted(0, None);
        run.heartbeat();
        run.observe(&AgentEvent::ToolStart {
            id: "t1".into(),
            name: "bash".into(),
            input: json!({}),
        });
        let kinds: Vec<_> = rec
            .snapshot()
            .iter()
            .map(|e| match e {
                RunEvent::Started { .. } => "started",
                RunEvent::Progress { .. } => "progress",
                RunEvent::Aborted { .. } => "aborted",
                _ => "other",
            })
            .collect();
        assert_eq!(kinds, ["started", "aborted"], "{kinds:?}");
    }

    #[test]
    fn heartbeat_skips_an_unchanged_sample() {
        let (rec, run) = rec_run();
        run.heartbeat();
        run.heartbeat();
        assert_eq!(
            rec.snapshot()
                .iter()
                .filter(|e| matches!(e, RunEvent::Progress { .. }))
                .count(),
            0,
            "empty in-flight sample must not POST on top of started: {:#?}",
            rec.snapshot()
        );

        run.observe(&AgentEvent::ToolStart {
            id: "t1".into(),
            name: "bash".into(),
            input: json!({}),
        });
        run.heartbeat();
        run.heartbeat();
        let progress: Vec<_> = rec
            .snapshot()
            .into_iter()
            .filter(|e| matches!(e, RunEvent::Progress { .. }))
            .collect();
        assert_eq!(
            progress.len(),
            1,
            "timer must not re-POST an unchanged sample: {progress:#?}"
        );
        match &progress[0] {
            RunEvent::Progress { tools, steps, .. } => {
                assert_eq!(tools, &["bash".to_string()]);
                assert_eq!(*steps, 0);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn turn_end_emits_steps() {
        let (rec, run) = rec_run();
        run.observe(&AgentEvent::TurnEnd {
            stop_reason: agent_core::StopReason::EndTurn,
            step: 3,
        });
        let progress: Vec<_> = rec
            .snapshot()
            .into_iter()
            .filter_map(|e| match e {
                RunEvent::Progress { steps, tools, .. } => Some((steps, tools)),
                _ => None,
            })
            .collect();
        assert_eq!(progress, [(3, Vec::new())], "{progress:?}");
    }

    #[test]
    fn heartbeat_interval_is_none_when_disabled() {
        assert!(heartbeat_interval(Duration::ZERO).is_none());
        assert_eq!(HEARTBEAT, Duration::from_secs(10));
    }

    #[tokio::test]
    async fn heartbeat_interval_is_some_when_enabled() {
        assert!(heartbeat_interval(HEARTBEAT).is_some());
    }

    #[test]
    fn progress_carries_active_tools_and_todos() {
        let (rec, run) = rec_run();
        run.observe(&AgentEvent::ToolStart {
            id: "a".into(),
            name: "bash".into(),
            input: json!({}),
        });
        run.observe(&AgentEvent::ToolStart {
            id: "b".into(),
            name: "grep".into(),
            input: json!({}),
        });
        run.observe(&AgentEvent::ToolProgress {
            id: "c".into(),
            name: crate::tools::todo::NAME.into(),
            snapshot: String::new(),
            details: Some(json!({ "todos": [{ "content": "ship", "status": "in_progress" }] })),
        });
        run.observe(&AgentEvent::ToolEnd {
            id: "a".into(),
            name: "bash".into(),
            result: String::new(),
            is_error: false,
        });
        let progress: Vec<_> = rec
            .snapshot()
            .into_iter()
            .filter_map(|e| match e {
                RunEvent::Progress { tools, todos, .. } => Some((tools, todos)),
                _ => None,
            })
            .collect();
        assert!(
            progress
                .iter()
                .any(|(tools, _)| tools == &["bash".to_string(), "grep".to_string()]),
            "concurrent tools must both appear: {progress:#?}"
        );
        assert!(
            progress
                .iter()
                .any(|(tools, _)| tools == &["grep".to_string()]),
            "ended tool must leave the sample: {progress:#?}"
        );
        assert!(
            progress.iter().any(|(_, todos)| todos.is_some()),
            "todos ride on the sample: {progress:#?}"
        );
    }

    #[test]
    fn summary_is_text_only_and_truncated_on_a_scalar_boundary() {
        let mut session = Session::new();
        session.push(Message::assistant(vec![
            ContentBlock::Thinking {
                text: "secret chain of thought".into(),
                signature: "sig".into(),
            },
            ContentBlock::RedactedThinking { data: "enc".into() },
            ContentBlock::text("visible"),
        ]));
        assert_eq!(closing_summary(&session).as_deref(), Some("visible"));

        let cut = truncate_chars("ééé", 2);
        assert_eq!(cut, "éé");
        assert!(cut.is_char_boundary(cut.len()));
        let emoji = truncate_chars("😀😀😀", 1);
        assert_eq!(emoji.chars().count(), 1);
    }

    #[test]
    fn command_id_is_echoed_when_present() {
        let (rec, run) = rec_run();
        run.succeeded(1, false, None, None);
        match &rec.snapshot()[0] {
            RunEvent::Started {
                command_id: Some(id),
                session_id,
                ..
            } => {
                assert_eq!(id, "cmd-1");
                assert_eq!(session_id, "sess-1");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn unconfigured_open_is_disabled() {
        let sink = open(None, &[]).unwrap();
        assert!(!sink.enabled());
        let sink = open(Some(""), &[]).unwrap();
        assert!(!sink.enabled());
    }

    #[test]
    fn malformed_url_fails_at_construction() {
        assert!(open(Some("file:///etc/passwd"), &[]).is_err());
        assert!(open(Some("not a url"), &[]).is_err());
        assert!(parse_endpoint("http://127.0.0.1:9/ok").is_ok());
        assert!(open(Some("http://127.0.0.1:9/ok"), &["no-colon".into()]).is_err());
        let unix_err = parse_endpoint("unix://").unwrap_err();
        assert!(
            unix_err.contains("unix") || unix_err.contains("socket path"),
            "{unix_err}"
        );
        assert!(parse_endpoint("unix:///tmp/lifecycle.sock").is_ok());
        assert!(parse_endpoint("unix://localhost/tmp/lifecycle.sock").is_ok());
        assert!(parse_endpoint("unix://example.com/tmp/lifecycle.sock").is_err());
        assert!(parse_endpoint("unix:///tmp/lifecycle.sock?x=1").is_err());
    }

    async fn serve_collector_conn<S>(
        mut sock: S,
        delay: Duration,
        status: u16,
        seen: Arc<Mutex<Vec<(String, String)>>>,
        hits: Arc<AtomicUsize>,
    ) where
        S: AsyncRead + AsyncWrite + Unpin,
    {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        let (mut need, mut head_end) = (0usize, None);
        loop {
            let Ok(n) = sock.read(&mut chunk).await else {
                return;
            };
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&chunk[..n]);
            if head_end.is_none()
                && let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n")
            {
                head_end = Some(p + 4);
                let head = String::from_utf8_lossy(&buf[..p]).to_lowercase();
                need = head
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0);
            }
            if let Some(h) = head_end
                && buf.len() >= h + need
            {
                break;
            }
        }
        hits.fetch_add(1, Ordering::Relaxed);
        let head_end = head_end.unwrap_or(buf.len());
        let headers = String::from_utf8_lossy(&buf[..head_end]).into_owned();
        let body = String::from_utf8_lossy(&buf[head_end..]).into_owned();
        lock(&seen).push((headers, body));
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        let resp =
            format!("HTTP/1.1 {status} OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok");
        let _ = sock.write_all(resp.as_bytes()).await;
        let _ = sock.shutdown().await;
    }

    async fn collect_posts(
        delay: Duration,
        status: u16,
        seen: Arc<Mutex<Vec<(String, String)>>>,
        hits: Arc<AtomicUsize>,
    ) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((sock, _)) = listener.accept().await else {
                    return;
                };
                let seen = Arc::clone(&seen);
                let hits = Arc::clone(&hits);
                tokio::spawn(async move {
                    serve_collector_conn(sock, delay, status, seen, hits).await;
                });
            }
        });
        format!("http://{addr}/lifecycle")
    }

    #[cfg(unix)]
    async fn collect_unix_posts(
        delay: Duration,
        status: u16,
        seen: Arc<Mutex<Vec<(String, String)>>>,
        hits: Arc<AtomicUsize>,
    ) -> (String, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let sock_path = dir.path().join("lifecycle.sock");
        let listener = tokio::net::UnixListener::bind(&sock_path).unwrap();
        let url = format!("unix://{}", sock_path.display());
        tokio::spawn(async move {
            loop {
                let Ok((sock, _)) = listener.accept().await else {
                    return;
                };
                let seen = Arc::clone(&seen);
                let hits = Arc::clone(&hits);
                tokio::spawn(async move {
                    serve_collector_conn(sock, delay, status, seen, hits).await;
                });
            }
        });
        (url, dir)
    }

    #[tokio::test]
    async fn http_posts_started_then_terminal() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let hits = Arc::new(AtomicUsize::new(0));
        let url = collect_posts(Duration::ZERO, 200, seen.clone(), hits.clone()).await;
        let sink = open(Some(&url), &["X-Tenant: acme".into()]).unwrap();
        let run = Run::begin(Arc::clone(&sink), "s", Some("p1".into()), "claude-test");
        run.succeeded(2, false, Some("hello".into()), None);
        run.drain().await;
        let bodies: Vec<Value> = lock(&seen)
            .iter()
            .filter_map(|(_, b)| serde_json::from_str(b).ok())
            .collect();
        assert!(bodies.iter().any(|v| v["type"] == "started"), "{bodies:#?}");
        assert!(
            bodies
                .iter()
                .any(|v| v["type"] == "succeeded" && v["summary"] == "hello"),
            "{bodies:#?}"
        );
        let headers = lock(&seen)[0].0.to_lowercase();
        assert!(headers.contains("x-tenant: acme"), "{headers}");
        assert_eq!(bodies.iter().filter(|v| v["type"] == "started").count(), 1);
        assert_eq!(
            bodies.iter().filter(|v| v["type"] == "succeeded").count(),
            1
        );
    }

    #[tokio::test]
    async fn terminated_is_evicted_after_terminal_flush() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let hits = Arc::new(AtomicUsize::new(0));
        // Hold each POST so a late progress emit still sees `terminated` before flush.
        let url = collect_posts(Duration::from_millis(50), 200, seen.clone(), hits.clone()).await;
        let http = Arc::new(HttpLifecycle::spawn(&url, vec![]).unwrap());
        for i in 0..3 {
            let run = Run::begin(http.clone(), "s", None, "m");
            let run_id = run.run_id().to_string();
            run.succeeded(1, false, Some(format!("done-{i}")), None);
            http.emit(RunEvent::Progress {
                run_id,
                session_id: "s".into(),
                command_id: None,
                ts: 0,
                steps: 99,
                tools: vec!["bash".into()],
                attempt: None,
                todos: None,
                depth: None,
            });
            http.drain().await;
            assert_eq!(
                lock(&http.inner.terminated).len(),
                0,
                "run {i}: terminated must not accumulate after flush"
            );
        }
        let bodies: Vec<Value> = lock(&seen)
            .iter()
            .filter_map(|(_, b)| serde_json::from_str(b).ok())
            .collect();
        assert_eq!(
            bodies.iter().filter(|v| v["type"] == "progress").count(),
            0,
            "late progress must not ride out after terminal: {bodies:#?}"
        );
        assert_eq!(
            bodies.iter().filter(|v| v["type"] == "succeeded").count(),
            3,
            "{bodies:#?}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unix_socket_posts_started_then_terminal() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let hits = Arc::new(AtomicUsize::new(0));
        let (url, _dir) = collect_unix_posts(Duration::ZERO, 200, seen.clone(), hits.clone()).await;
        let sink = open(Some(&url), &[]).unwrap();
        let run = Run::begin(Arc::clone(&sink), "s", Some("p1".into()), "claude-test");
        run.succeeded(1, false, Some("hello".into()), None);
        run.drain().await;
        let bodies: Vec<Value> = lock(&seen)
            .iter()
            .filter_map(|(_, b)| serde_json::from_str(b).ok())
            .collect();
        assert!(bodies.iter().any(|v| v["type"] == "started"), "{bodies:#?}");
        assert!(
            bodies
                .iter()
                .any(|v| v["type"] == "succeeded" && v["summary"] == "hello"),
            "{bodies:#?}"
        );
    }

    #[tokio::test]
    async fn a_slow_consumer_does_not_block_emit() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let hits = Arc::new(AtomicUsize::new(0));
        let url = collect_posts(Duration::from_secs(3), 200, seen, hits).await;
        let sink = open(Some(&url), &[]).unwrap();
        let run = Run::begin(Arc::clone(&sink), "s", None, "m");
        let start = std::time::Instant::now();
        run.succeeded(1, false, None, None);
        assert!(
            start.elapsed() < Duration::from_millis(200),
            "emit awaited the consumer ({:?})",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn progress_is_latest_wins_not_a_log() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let hits = Arc::new(AtomicUsize::new(0));
        // Hold the first POST (started) long enough that several progress samples coalesce.
        let url = collect_posts(Duration::from_millis(80), 200, seen.clone(), hits).await;
        let sink = open(Some(&url), &[]).unwrap();
        let run = Run::begin(Arc::clone(&sink), "s", None, "m");
        for name in ["read", "edit", "bash"] {
            run.observe(&AgentEvent::ToolStart {
                id: name.into(),
                name: name.into(),
                input: json!({}),
            });
        }
        run.succeeded(4, false, None, None);
        run.drain().await;
        let bodies: Vec<Value> = lock(&seen)
            .iter()
            .filter_map(|(_, b)| serde_json::from_str(b).ok())
            .collect();
        let progress: Vec<_> = bodies.iter().filter(|v| v["type"] == "progress").collect();
        // Latest-wins means we must not POST every sample. One or two (a sample that raced
        // started's POST, plus the coalesced remainder) is fine; three is the un-coalesced log.
        assert!(
            progress.len() <= 2,
            "progress should coalesce, got {}: {bodies:#?}",
            progress.len()
        );
        assert!(
            bodies.iter().any(|v| v["type"] == "succeeded"),
            "{bodies:#?}"
        );
    }
}
