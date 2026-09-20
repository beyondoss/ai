//! MCP (Model Context Protocol) client support — this crate's extension mechanism.
//!
//! pi (the TypeScript reference this codebase tracks) has its own in-process extension system
//! (`registerTool`/`registerCommand` modules loaded directly into the host). Beyond deliberately does
//! *not* copy that shape: MCP is a standardized, language-agnostic protocol (JSON-RPC over stdio or
//! HTTP) with an existing server ecosystem (filesystem, GitHub, Slack, Postgres, ...), and — unlike
//! loading `.so`/`.dylib` plugins via `libloading` — needs zero `unsafe` code, which this workspace
//! forbids (`unsafe_code = "forbid"`).
//!
//! [`connect_all`] connects to every [`McpServerConfig`](crate::settings::McpServerConfig) configured
//! in `settings.json` (global or a trusted project's own), lists each server's tools via `tools/list`,
//! and wraps each into an [`McpTool`] — an ordinary [`agent_core::Tool`] registered into the same
//! [`agent_core::ToolRegistry`] the built-in tools use, so the model sees no difference between a
//! built-in `read`/`bash` and an MCP-discovered tool. Every tool is namespaced
//! `mcp__<server-name>__<tool-name>` (the same convention Claude Code itself uses), so it can never
//! collide with a built-in tool, and a collision with another server's tool is scoped to that one
//! server's own name.
//!
//! Connecting happens exactly once, at startup (`main.rs`'s `run` path, and once before `serve`'s main
//! loop) — not re-done on a `serve` registry rebuild (`set_model`/`set_thinking` reuse the
//! already-connected tools; see `serve.rs::ServeConfig::mcp_tools`). A server that fails to connect is
//! skipped with a warning rather than failing the whole agent's startup (fail-soft).
//!
//! **Progress.** `tools/call` requests carry an MCP `progressToken` (rmcp injects one on every
//! request). When the server emits `notifications/progress`, [`McpTool::run_streaming`] forwards each
//! update into the harness [`ToolProgress`] sink — the same `AgentEvent::ToolProgress` path bash/todo
//! already use — so a long MCP call is observably live rather than a silent hang until the final
//! result.
//!
//! Sampling and roots client capabilities remain advertised for protocol completeness even though
//! SEP-2577 deprecates them; see [`crate::tools::mcp_host`].

#![expect(deprecated)] // Sampling/roots: SEP-2577-deprecated but still on the wire.
//!
//! **Session enablement.** Configured servers stay connected (or lazily dormant) for the process, but
//! which ones are *advertised* is session-scoped via [`McpEnabledSet`] (`serve`'s `set_mcp_enabled`).
//! That is the kit-shaping seam: disable defaults with `--tools`/`--exclude-tools`, then enable only
//! the MCP servers this task needs — without reconnecting or restarting.
//!
//! **Resources / prompts.** At connect we also `resources/list` and `prompts/list`, wrapping each as
//! an ordinary tool (`mcp__<server>__resource__<name>`, `mcp__<server>__prompt__<name>`) so the model
//! can read resources and expand prompts without a separate host protocol. Completions are host-facing
//! via `serve`'s `mcp_complete` RPC (argument autocomplete is a UI concern, not a model tool).
//!
//! **Elicitation / sampling / MRTR / tasks.** [`McpHandler`] advertises those client capabilities and
//! fulfills server→client requests through [`crate::tools::mcp_host`] hubs (`serve` installs UI gates;
//! headless `run` declines / rejects). Tool calls drive `call_tool_once` so SEP-2322 `input_required`
//! rounds and SEP-2663 task handles complete under protocol `2026-07-28`. Connect uses
//! [`ClientLifecycleMode::Auto`] preferring `2026-07-28` (discover), with legacy initialize fallback.
//!
//! Adding a brand-new server config mid-process (vs enabling one already configured at startup) remains
//! out of scope.
//!
//! **Tasks (SEP-2663).** Client advertises `io.modelcontextprotocol/tasks`. When `tools/call` returns
//! `resultType: "task"`, we poll `tasks/get` (honoring `pollIntervalMs`), fulfill in-task
//! `inputRequests` via `tasks/update`, surface `statusMessage` as [`ToolProgress`], and best-effort
//! `tasks/cancel` if the tool future is dropped mid-poll.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_core::{ImageSource, Tool, ToolError, ToolOutput, ToolProgress};
use async_trait::async_trait;
use http::{HeaderName, HeaderValue};
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, CancelTaskParams, ClientCapabilities,
    ClientInfo, ContentBlock, CreateMessageRequestParams, CreateMessageResult, CreateTaskResult,
    DEFAULT_MRTR_MAX_ROUNDS, ElicitRequestParams, ElicitResult, GetPromptRequestParams,
    GetTaskParams, Implementation, InputRequest, InputRequests, InputResponses, ListRootsResult,
    ProgressNotificationParam, ProgressToken, ProtocolVersion, ReadResourceRequestParams,
    ResourceContents, TaskPayload, TaskStatus, UpdateTaskParams,
};
use rmcp::service::{
    ClientLifecycleMode, ClientServiceExt, NotificationContext, Peer, RequestContext,
    RunningService,
};
use rmcp::transport::streamable_http_client::{
    StreamableHttpClientTransport, StreamableHttpClientTransportConfig,
};
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use rmcp::{ClientHandler, ErrorData as McpError, RoleClient};
use serde_json::{Map, Value, json};

use crate::settings::{McpServerConfig, McpTransport};
use crate::tools::mcp_host::{ElicitationAsk, McpHost};

/// Process-scoped host callbacks (elicitation / sampling) for the servers connected **once, at
/// startup** from the operator's own settings: those connections are shared by every session, so
/// there is no one session to route a server's question to.
///
/// A session that dials its *own* connectors (service mode — see [`connect_granted`]) passes its own
/// [`McpHost`] instead, and a question from one of those reaches the session that asked.
pub fn host() -> Arc<McpHost> {
    static HOST: std::sync::OnceLock<Arc<McpHost>> = std::sync::OnceLock::new();
    HOST.get_or_init(|| Arc::new(McpHost::new())).clone()
}

/// An HTTP header whose value is a credential, **taken exactly as given**.
///
/// The settings path resolves a configured `headers` value through
/// [`resolve_config_value`](crate::settings) first — a `!command` runs a shell command and `$VAR`
/// reads the process environment. That is right for a value an operator typed into their own
/// `settings.json`, and catastrophic for one that arrived over the wire: a session grant's
/// `!echo …` header would execute on the replica, and a `$AWS_SECRET_ACCESS_KEY` one would exfiltrate
/// the replica's environment to whoever minted the grant. A `SecretHeader` never goes near that
/// resolution, and `Debug` shows only the name.
#[derive(Clone)]
pub struct SecretHeader {
    name: HeaderName,
    value: HeaderValue,
}

impl SecretHeader {
    /// Validate one pre-resolved header. The error names the header, never its value.
    pub fn new(name: &str, value: &str) -> Result<Self, String> {
        let header = HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| format!("`{name}` is not a valid header name"))?;
        let mut value = HeaderValue::from_str(value)
            .map_err(|_| format!("the value for `{name}` is not a valid header value"))?;
        // Keeps it out of hyper's HPACK index and out of `HeaderValue`'s own `Debug`.
        value.set_sensitive(true);
        Ok(Self {
            name: header,
            value,
        })
    }
}

impl std::fmt::Debug for SecretHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SecretHeader({}: ***)", self.name)
    }
}

/// A grant connector's pre-resolved HTTP dial: the process-wide client it goes out on, and its own
/// credential headers. Kept on the [`McpConnection`] so a redial after an idle reap is identical to
/// the first dial — the grant that carried these is long out of scope by then.
#[derive(Clone, Debug)]
struct HttpDial {
    /// The daemon's `mcp_http` client: ALPN, the `web` tool's SSRF resolver, and deliberately not
    /// the gateway's h2c pool — a tenant's connector is not the gateway.
    client: reqwest::Client,
    headers: Vec<SecretHeader>,
}

/// Everything about dialing one server that isn't in its [`McpServerConfig`]: which host hub its
/// server→client requests reach, and — for a grant connector — how to reach it.
#[derive(Clone)]
struct Dial {
    host: Arc<McpHost>,
    /// `None` for a settings-configured server: it resolves its own headers and may carry an OAuth
    /// bearer token from this host's store. A grant connector has neither.
    http: Option<HttpDial>,
}

impl Default for Dial {
    fn default() -> Self {
        Self {
            host: host(),
            http: None,
        }
    }
}

/// A connector URL with its query string stripped, for an error or a log line. A grant's connector
/// URL is the control plane's to shape and can carry a credential in a query parameter.
fn redact_url(url: &str) -> String {
    match url.split_once('?') {
        Some((head, _)) => format!("{head}?<redacted>"),
        None => url.to_owned(),
    }
}

fn client_lifecycle() -> ClientLifecycleMode {
    // Prefer `server/discover` so peers that speak `2026-07-28` negotiate MRTR (SEP-2322). Auto
    // falls back to legacy `initialize` within 10s when discover is unsupported (-32601) or times
    // out — our fixture answers discover so the fallback is never taken in tests.
    ClientLifecycleMode::Auto {
        preferred_versions: vec![ProtocolVersion::V_2026_07_28],
        legacy_version: Some(ProtocolVersion::V_2025_11_25),
    }
}

fn client_info() -> ClientInfo {
    ClientInfo::new(
        ClientCapabilities::builder()
            .enable_elicitation()
            .enable_sampling()
            .enable_roots()
            .enable_tasks()
            .build(),
        Implementation::new("beyond-ai-agent", env!("CARGO_PKG_VERSION")),
    )
    .with_protocol_version(ProtocolVersion::V_2026_07_28)
}

/// Routes progress, elicitation, and sampling for one MCP connection.
///
/// Tool calls use [`RunningService::call_tool_once`] (MRTR + SEP-2663 tasks). rmcp injects a
/// `progressToken` we do not observe. Token-keyed sinks still work when a token is known; otherwise
/// progress falls back to the LIFO active sink pushed for the in-flight call (covers the normal
/// model path). Task `statusMessage` updates also emit on that sink while polling.
#[derive(Clone)]
struct McpHandler {
    server_name: String,
    /// Where this connection's server→client requests (elicitation, sampling) go. Held per
    /// connection rather than read from a process-wide global, so a session that dialed its own
    /// connectors answers their questions itself instead of whichever session installed a gate last.
    host: Arc<McpHost>,
    sinks: Arc<std::sync::Mutex<HashMap<ProgressToken, ToolProgress>>>,
    /// In-flight tool-call progress sinks (LIFO). See [`Self::push_active`].
    active: Arc<std::sync::Mutex<Vec<ToolProgress>>>,
}

impl McpHandler {
    fn new(server_name: impl Into<String>, host: Arc<McpHost>) -> Self {
        Self {
            server_name: server_name.into(),
            host,
            sinks: Arc::new(std::sync::Mutex::new(HashMap::new())),
            active: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    /// Where a server→client elicitation goes: **this connection's** host, never a process-wide
    /// one. Split out of the `ClientHandler` method so the routing is testable without standing up
    /// an rmcp `RequestContext`.
    async fn elicit(&self, params: ElicitRequestParams) -> ElicitResult {
        self.host
            .elicitation
            .elicit(ElicitationAsk {
                server: self.server_name.clone(),
                params,
            })
            .await
    }

    fn push_active(&self, progress: ToolProgress) -> ActiveProgressGuard {
        if let Ok(mut active) = self.active.lock() {
            active.push(progress);
        }
        ActiveProgressGuard {
            active: self.active.clone(),
        }
    }
}

/// Pops the active progress sink pushed for one `call_tool` when the call ends.
struct ActiveProgressGuard {
    active: Arc<std::sync::Mutex<Vec<ToolProgress>>>,
}

impl Drop for ActiveProgressGuard {
    fn drop(&mut self) {
        if let Ok(mut active) = self.active.lock() {
            let _ = active.pop();
        }
    }
}

impl ClientHandler for McpHandler {
    fn get_info(&self) -> ClientInfo {
        client_info()
    }

    async fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        let sink = self
            .sinks
            .lock()
            .ok()
            .and_then(|sinks| sinks.get(&params.progress_token).cloned())
            .or_else(|| {
                self.active
                    .lock()
                    .ok()
                    .and_then(|active| active.last().cloned())
            });
        let Some(progress) = sink else {
            return;
        };
        let snapshot = format_progress_snapshot(&params);
        let details = progress_details(&params);
        progress.emit(snapshot, Some(details));
    }

    async fn create_elicitation(
        &self,
        request: ElicitRequestParams,
        _context: RequestContext<RoleClient>,
    ) -> Result<ElicitResult, McpError> {
        Ok(self.elicit(request).await)
    }

    async fn create_message(
        &self,
        params: CreateMessageRequestParams,
        _context: RequestContext<RoleClient>,
    ) -> Result<CreateMessageResult, McpError> {
        self.host.sampling.create_message(params).await
    }
}

fn format_progress_snapshot(params: &ProgressNotificationParam) -> String {
    match (&params.message, params.total) {
        (Some(message), Some(total)) => {
            format!("{message} ({}/{})", params.progress as i64, total as i64)
        }
        (Some(message), None) => message.clone(),
        (None, Some(total)) => format!("{}/{}", params.progress as i64, total as i64),
        (None, None) => format!("{}", params.progress as i64),
    }
}

fn progress_details(params: &ProgressNotificationParam) -> Value {
    json!({
        "progress": params.progress,
        "total": params.total,
        "message": params.message,
    })
}

/// A connected MCP server's live client handle. Shared (via `Arc`) by every tool the server produced.
type McpClient = RunningService<RoleClient, McpHandler>;

/// Session-scoped gate over which configured MCP servers' tools are advertised to the model.
///
/// Configured servers stay connected (or lazily dormant) for the whole process; this only controls
/// which ones enter the [`ToolRegistry`] on each rebuild. `None` means every configured server is
/// enabled (the default, matching prior behavior).
#[derive(Clone, Default)]
pub struct McpEnabledSet {
    inner: Arc<std::sync::Mutex<Option<HashSet<String>>>>,
}

impl McpEnabledSet {
    /// Every configured server enabled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the allow-list. `None` = all enabled; `Some(empty)` = none; `Some({a,b})` = only those.
    pub fn set(&self, enabled: Option<HashSet<String>>) {
        if let Ok(mut guard) = self.inner.lock() {
            *guard = enabled;
        }
    }

    /// Current allow-list snapshot (`None` = all enabled).
    pub fn snapshot(&self) -> Option<HashSet<String>> {
        self.inner.lock().ok().and_then(|g| g.clone())
    }

    /// Whether tools from `server` should be advertised.
    pub fn allows(&self, server: &str) -> bool {
        match self.snapshot() {
            None => true,
            Some(set) => set.contains(server),
        }
    }
}

/// The prefix every MCP-discovered tool's registered name carries — `mcp__<server>__<tool>` — so it
/// can never collide with a built-in tool. Matches the convention Claude Code itself uses for the
/// identical problem.
fn registered_name(server: &str, remote_tool: &str) -> String {
    format!("mcp__{server}__{remote_tool}")
}

fn registered_resource_name(server: &str, resource_name: &str) -> String {
    format!("mcp__{server}__resource__{resource_name}")
}

fn registered_prompt_name(server: &str, prompt_name: &str) -> String {
    format!("mcp__{server}__prompt__{prompt_name}")
}

/// Inverse of [`registered_name`]: `mcp__filesystem__read_file` → `Some("filesystem")`.
pub fn server_name_from_registered(tool_name: &str) -> Option<&str> {
    let rest = tool_name.strip_prefix("mcp__")?;
    let (server, _) = rest.split_once("__")?;
    if server.is_empty() {
        None
    } else {
        Some(server)
    }
}

/// Keep only MCP tools whose server is currently enabled.
pub fn filter_by_enabled(tools: &[Arc<dyn Tool>], enabled: &McpEnabledSet) -> Vec<Arc<dyn Tool>> {
    tools
        .iter()
        .filter(|t| match server_name_from_registered(t.name()) {
            Some(server) => enabled.allows(server),
            None => true,
        })
        .cloned()
        .collect()
}

/// Distinct server names represented in a tool list, sorted — for `get_mcp` / diagnostics.
pub fn server_names_from_tools(tools: &[Arc<dyn Tool>]) -> Vec<String> {
    let mut names: Vec<String> = tools
        .iter()
        .filter_map(|t| server_name_from_registered(t.name()).map(str::to_string))
        .collect();
    names.sort();
    names.dedup();
    names
}

/// One MCP-discovered tool. Holds no local behavior at all — `run` forwards straight to the remote
/// server's `tools/call`; every byte of actual behavior lives on the other end of `client`.
struct McpTool {
    /// Registered/advertised name: `mcp__<server>__<tool>` (see [`registered_name`]).
    name: String,
    description: String,
    input_schema: Value,
    /// The bare tool name as the *server* knows it (unprefixed) — what actually goes out over
    /// `tools/call`.
    remote_name: String,
    /// For error messages only, so a failure names which server misbehaved.
    server_name: String,
    /// The server's connection, which may or may not currently have a live process behind it — see
    /// [`McpConnection`]. Shared by every tool discovered from the same server, so reaping one reaps
    /// them all and reconnecting serves them all.
    conn: Arc<McpConnection>,
}

/// A connection to one MCP server that can be dropped and rebuilt underneath the tools using it.
///
/// The point is memory. An MCP server is a whole language runtime sitting on a guest waiting to be
/// asked something: measured on the vps primitive, `@playwright/mcp` costs **63.9 MB of anonymous
/// memory** while completely idle, and 66.7 MB of that is one `require("playwright-core")` — a
/// browser API loaded before any browser exists. On a guest whose whole job is an agent, that was 87%
/// of everything anonymous in the VM.
///
/// Tools are still *discovered* eagerly at startup, because the model has to be told what it can call
/// before it calls anything. But discovery is the only thing that needs a live process: an
/// [`McpTool`] owns its own name, description, and schema, so once it exists the process behind it is
/// dead weight until someone actually calls it. So the process is reaped after
/// [`IDLE_REAP_AFTER`] without a call, and re-spawned on the next one.
///
/// Dropping the client is what kills the child: rmcp's `ChildWithCleanup` reaps the process in its
/// `Drop`. There is no separate shutdown to call, and no zombie left behind.
/// A connected server: the client, and the process group to sweep when it goes away.
///
/// The group is kept beside the client rather than on the connection because it belongs to *this*
/// process — a reconnect after a reap starts a new one, and sweeping the old group id then would
/// either do nothing or, if the id had been recycled, kill something unrelated.
struct Live {
    client: Arc<McpClient>,
    /// `None` for HTTP transports: there is no process of ours to reap.
    pgid: Option<u32>,
}

struct McpConnection {
    config: McpServerConfig,
    /// How this server was dialed the first time — its host hub, and (for a grant connector) the
    /// client and credential headers. Kept so a redial after a reap cannot drift from it.
    dial: Dial,
    /// `None` once reaped (or before the first reconnect). A `tokio::sync::Mutex` rather than a
    /// `std` one because reconnecting is `await`-ing I/O while holding it — two concurrent tool calls
    /// arriving on a reaped connection must produce one process, not two.
    client: tokio::sync::Mutex<Option<Live>>,
    /// When the connection was last used, for the reaper. Seconds since the process started, so it
    /// fits an atomic and needs no lock on the hot path.
    last_used: std::sync::atomic::AtomicU64,
    /// How long *this* connection may idle before its process is reaped; `ZERO` never reaps it.
    ///
    /// Per-connection rather than a process-wide setting the reaper captured once: the sweeper is
    /// started lazily by whoever connects first, so a captured window would silently apply to every
    /// server configured afterwards — including one that asked never to be reaped. A test caught
    /// exactly that.
    idle_after: Duration,
}

/// How long a server may sit unused before its process is reaped.
///
/// Short enough that a guest which boots and is never asked to browse gives the memory back promptly,
/// long enough that a working session doesn't pay a re-spawn between consecutive tool calls. A
/// re-spawn costs a process start plus an MCP handshake — a second or two for a heavy server — which
/// is noise against the model round trip that precedes every tool call.
///
/// `Duration::ZERO` disables reaping entirely — every configured server stays resident for the
/// process's life, which is the behavior that existed before this.
pub const DEFAULT_IDLE_REAP_AFTER: Duration = Duration::from_secs(120);

/// The idle window after which an unused MCP server's process is reaped, re-spawned on the next
/// call: [`DEFAULT_IDLE_REAP_AFTER`], or whatever `BEYOND_AI_AGENT_MCP_IDLE_SECS` says (`0` disables
/// reaping and keeps every server resident for the process's life).
///
/// Read here rather than in one binary's argument parsing, so the `run` path, the daemon's own
/// configured servers, and a service session's grant connectors all honour the same knob.
pub fn idle_reap_after_from_env() -> Duration {
    std::env::var("BEYOND_AI_AGENT_MCP_IDLE_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_IDLE_REAP_AFTER)
}

/// Every live connection, weakly. The reaper sweeps this rather than owning the connections, so a
/// connection disappears from it as soon as the tools holding it are dropped — a `serve` registry
/// rebuild or a finished `run` never leaves the reaper keeping a server alive.
static LIVE: std::sync::Mutex<Vec<std::sync::Weak<McpConnection>>> =
    std::sync::Mutex::new(Vec::new());

/// Process start, the epoch `last_used` counts from. A monotonic seconds counter rather than
/// `SystemTime`, so a clock step can never make an idle server look freshly used (or vice versa).
static STARTED: std::sync::LazyLock<Instant> = std::sync::LazyLock::new(Instant::now);

fn now_secs() -> u64 {
    STARTED.elapsed().as_secs()
}

/// How often the reaper looks: half the *shortest* live window, capped at 30s and floored at 1s.
///
/// Recomputed every pass rather than fixed when the sweeper starts. The sweeper is process-wide and
/// starts once, so a period taken from whichever connection happened to register first would be wrong
/// for every server configured afterwards — a 60s server registering ahead of a 1s one would leave the
/// 1s one un-swept for half a minute. A test caught exactly that.
///
/// Half the window, so a server is reaped within roughly it rather than up to twice it. The floor
/// keeps a tiny configured window from spinning (and from handing a sleep a zero period).
fn reap_tick(windows: impl Iterator<Item = Duration>) -> Duration {
    windows
        .filter(|w| !w.is_zero())
        .min()
        .unwrap_or(DEFAULT_IDLE_REAP_AFTER)
        .div_f32(2.0)
        .min(Duration::from_secs(30))
        .max(Duration::from_secs(1))
}

/// Whether a sweeper is currently running.
///
/// Deliberately not a `std::sync::Once`. The sweeper is a tokio task, so it lives and dies with the
/// runtime that spawned it, and a runtime can go away underneath it — every `#[tokio::test]` builds
/// its own, and an embedder may build one per unit of work. With a `Once`, the first runtime to shut
/// down would take the sweeper with it and nothing could ever start another: every server configured
/// after that point would stay resident forever, which is precisely the leak this module exists to
/// prevent, made invisible. The task clears this flag as it is dropped, so the next connection that
/// wants reaping starts a fresh sweeper.
static REAPER_ALIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Wakes the sweeper when a connection registers.
///
/// The cadence comes from the live set, so a pass that ran before a connection existed slept on a
/// cadence computed without it: register a 120s server, then a 1s one, and the 1s one goes unswept for
/// the first 30 seconds of its life. Rather than poll fast enough to make that invisible — which costs
/// wakeups forever to fix a moment — the registration says so.
static REAPER_WAKE: std::sync::LazyLock<tokio::sync::Notify> =
    std::sync::LazyLock::new(tokio::sync::Notify::new);

/// Clears [`REAPER_ALIVE`] when the sweeper task ends — including the case that matters, the task
/// being *dropped* by a shutting-down runtime rather than returning.
struct ReaperGuard;

impl Drop for ReaperGuard {
    fn drop(&mut self) {
        REAPER_ALIVE.store(false, std::sync::atomic::Ordering::Release);
    }
}

/// Start the sweeper if one isn't already running.
///
/// One task for every server rather than one per connection: the work is a handful of atomic loads on
/// a timer, and a task per MCP server would be its own small leak on a long-lived `serve` daemon that
/// rebuilds its registry.
fn spawn_reaper_if_needed(idle: Duration) {
    // Nothing to sweep for if this caller never wants reaping; another caller that does will start it.
    if idle.is_zero() {
        return;
    }
    if REAPER_ALIVE.swap(true, std::sync::atomic::Ordering::AcqRel) {
        return;
    }
    tokio::spawn(async move {
        let _alive = ReaperGuard;
        loop {
            // Snapshot and drop the lock before awaiting: `reap_if_idle` takes an async lock, and
            // holding a std Mutex across an await would be a deadlock waiting to happen.
            let conns: Vec<Arc<McpConnection>> = {
                let Ok(mut live) = LIVE.lock() else { return };
                live.retain(|w| w.strong_count() > 0);
                live.iter().filter_map(std::sync::Weak::upgrade).collect()
            };
            let next = reap_tick(conns.iter().map(|c| c.idle_after));
            for conn in conns {
                // Holding an `Arc<McpConnection>` here is harmless: the busy check inside looks at
                // the strong count of the *client*, which only a call in flight clones.
                conn.reap_if_idle().await;
            }
            // Whichever comes first: the cadence elapsing, or a new connection changing what the
            // cadence should be.
            tokio::select! {
                () = tokio::time::sleep(next) => {}
                () = REAPER_WAKE.notified() => {}
            }
        }
    });
}

/// Put a connection under the sweeper's eye. Weakly, so the connection disappearing (a `serve`
/// registry rebuild, a finished `run`) takes it off the list on its own.
fn register_for_reaping(conn: &Arc<McpConnection>, idle: Duration) {
    if let Ok(mut live) = LIVE.lock() {
        live.push(Arc::downgrade(conn));
    }
    spawn_reaper_if_needed(idle);
    // Only meaningful if a sweeper was already running; a fresh one reads the live set immediately.
    REAPER_WAKE.notify_one();
}

impl McpConnection {
    fn new(
        config: McpServerConfig,
        dial: Dial,
        client: McpClient,
        pgid: Option<u32>,
        idle_after: Duration,
    ) -> Self {
        Self {
            config,
            dial,
            client: tokio::sync::Mutex::new(Some(Live {
                client: Arc::new(client),
                pgid,
            })),
            last_used: std::sync::atomic::AtomicU64::new(now_secs()),
            idle_after,
        }
    }

    /// A connection with no process behind it yet, for tools rebuilt from a cached manifest. The
    /// first `tools/call` dials; a boot that never calls one never starts a server at all.
    fn dormant(config: McpServerConfig, dial: Dial, idle_after: Duration) -> Self {
        Self {
            config,
            dial,
            client: tokio::sync::Mutex::new(None),
            last_used: std::sync::atomic::AtomicU64::new(now_secs()),
            idle_after,
        }
    }

    /// The live client, connecting first if the process was reaped.
    async fn client(&self) -> Result<Arc<McpClient>, String> {
        self.last_used
            .store(now_secs(), std::sync::atomic::Ordering::Relaxed);
        let mut guard = self.client.lock().await;
        if let Some(live) = guard.as_ref() {
            return Ok(live.client.clone());
        }
        let (client, pgid) = connect_one_client(&self.config, &self.dial).await?;
        let client = Arc::new(client);
        *guard = Some(Live {
            client: client.clone(),
            pgid,
        });
        Ok(client)
    }

    /// Drop the process if it has been idle long enough. Returns whether it reaped.
    ///
    /// Never reaps while a call is in flight: an in-flight call holds an `Arc` clone of the client, so
    /// a strong count above one means someone is still using it and the reap is skipped. Without that
    /// check a long-running `browser_navigate` could have the process killed out from under it.
    async fn reap_if_idle(&self) -> bool {
        let after = self.idle_after;
        if after.is_zero() {
            return false;
        }
        let idle =
            now_secs().saturating_sub(self.last_used.load(std::sync::atomic::Ordering::Relaxed));
        if idle < after.as_secs() {
            return false;
        }
        let mut guard = self.client.lock().await;
        match guard.as_ref() {
            Some(live) if Arc::strong_count(&live.client) == 1 => {
                let pgid = live.pgid;
                // Drops the client, which closes the server's stdin and kills it...
                *guard = None;
                // ...and this takes anything it forked away from itself, which a kill aimed at the
                // server alone leaves running.
                sweep_process_group(pgid);
                tracing::debug!(
                    server = %self.config.name,
                    idle_secs = idle,
                    "reaped an idle MCP server process"
                );
                true
            }
            _ => false,
        }
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Value {
        self.input_schema.clone()
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        // Non-streaming path (direct host invoke): still dials `tools/call`, just without registering
        // a ToolProgress sink. The model loop always uses `run_streaming`.
        self.call_remote(input, None).await
    }

    async fn run_streaming(
        &self,
        input: Value,
        progress: &ToolProgress,
    ) -> Result<ToolOutput, ToolError> {
        self.call_remote(input, Some(progress)).await
    }
}

impl McpTool {
    /// Shared `tools/call` path. Drives MRTR `input_required` and SEP-2663 task handles via
    /// [`drive_tool_call`]. When a [`ToolProgress`] sink is provided, it is pushed as the active
    /// fallback for progress notifications whose token we cannot observe, and for task
    /// `statusMessage` updates while polling.
    async fn call_remote(
        &self,
        input: Value,
        progress: Option<&ToolProgress>,
    ) -> Result<ToolOutput, ToolError> {
        let arguments = match input {
            Value::Object(map) => Some(map),
            Value::Null => None,
            other => {
                return Err(ToolError::InvalidInput(format!(
                    "expected a JSON object of arguments for `{}`, got: {other}",
                    self.name
                )));
            }
        };
        let mut params = CallToolRequestParams::new(self.remote_name.clone());
        if let Some(arguments) = arguments {
            params = params.with_arguments(arguments);
        }

        let client = self.conn.client().await.map_err(|e| {
            ToolError::Execution(format!(
                "mcp server `{}` is not reachable: {e}",
                self.server_name
            ))
        })?;

        let _active = progress.map(|p| client.service().push_active(p.clone()));
        let result = drive_tool_call(
            &client,
            &self.server_name,
            &self.remote_name,
            params,
            progress,
        )
        .await?;

        tool_output_from_result(&self.server_name, &self.remote_name, result)
    }
}

fn tool_call_err(server: &str, remote: &str, e: impl std::fmt::Display) -> ToolError {
    ToolError::Execution(format!(
        "mcp server `{server}` tool `{remote}` call failed: {e}"
    ))
}

/// Drive one `tools/call` through MRTR rounds and/or a SEP-2663 task lifecycle.
///
/// rmcp's high-level `call_tool` fulfills MRTR but errors on `CreateTaskResult`; with tasks
/// advertised we must use `call_tool_once` and poll ourselves. In-task elicitation/sampling/roots
/// reuse the same host hubs as nested and MRTR input (rmcp's fulfill helpers are private).
async fn drive_tool_call(
    client: &McpClient,
    server_name: &str,
    remote_name: &str,
    mut params: CallToolRequestParams,
    progress: Option<&ToolProgress>,
) -> Result<CallToolResult, ToolError> {
    for _round in 0..DEFAULT_MRTR_MAX_ROUNDS {
        match client
            .call_tool_once(params.clone())
            .await
            .map_err(|e| tool_call_err(server_name, remote_name, e))?
        {
            CallToolResponse::Complete(result) => return Ok(result),
            CallToolResponse::InputRequired(required) => {
                let responses = fulfill_input_requests(
                    &client.service().host,
                    server_name,
                    required.input_requests.unwrap_or_default(),
                )
                .await?;
                params.input_responses = (!responses.is_empty()).then_some(responses);
                params.request_state = required.request_state;
            }
            CallToolResponse::Task(create) => {
                return await_task(client, server_name, remote_name, create, progress).await;
            }
            other => {
                return Err(ToolError::Execution(format!(
                    "mcp server `{server_name}` tool `{remote_name}` returned unexpected tools/call response: {other:?}"
                )));
            }
        }
    }
    Err(ToolError::Execution(format!(
        "mcp server `{server_name}` tool `{remote_name}` exceeded {DEFAULT_MRTR_MAX_ROUNDS} input_required rounds"
    )))
}

/// Fulfill SEP-2322 / in-task `inputRequests` through the hubs of the host that owns this
/// connection — the session's own in service mode, the process-wide one otherwise.
async fn fulfill_input_requests(
    host: &McpHost,
    server_name: &str,
    requests: InputRequests,
) -> Result<InputResponses, ToolError> {
    let mut responses = BTreeMap::new();
    for (key, request) in requests {
        let value = match request {
            InputRequest::Elicitation(req) => {
                let result = host
                    .elicitation
                    .elicit(ElicitationAsk {
                        server: server_name.to_string(),
                        params: req.params,
                    })
                    .await;
                serde_json::to_value(result).map_err(|e| {
                    ToolError::Execution(format!("serialize elicitation result: {e}"))
                })?
            }
            InputRequest::CreateMessage(req) => {
                let result = host
                    .sampling
                    .create_message(req.params)
                    .await
                    .map_err(|e| ToolError::Execution(format!("MCP sampling failed: {e}")))?;
                serde_json::to_value(result)
                    .map_err(|e| ToolError::Execution(format!("serialize sampling result: {e}")))?
            }
            InputRequest::ListRoots(_) => serde_json::to_value(ListRootsResult::new(Vec::new()))
                .map_err(|e| ToolError::Execution(format!("serialize roots result: {e}")))?,
            other => {
                return Err(ToolError::Execution(format!(
                    "unsupported MCP input request variant while fulfilling `{key}`: {other:?}"
                )));
            }
        };
        responses.insert(key, value);
    }
    Ok(responses)
}

/// Best-effort `tasks/cancel` when the tool future is aborted mid-poll.
struct TaskCancelOnDrop {
    peer: Peer<RoleClient>,
    task_id: String,
    finished: bool,
}

impl TaskCancelOnDrop {
    fn finish(&mut self) {
        self.finished = true;
    }
}

impl Drop for TaskCancelOnDrop {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let peer = self.peer.clone();
        let task_id = self.task_id.clone();
        tokio::spawn(async move {
            let _ = peer.cancel_task(CancelTaskParams::new(task_id)).await;
        });
    }
}

/// Poll `tasks/get` until terminal; fulfill in-task input via `tasks/update`.
async fn await_task(
    client: &McpClient,
    server_name: &str,
    remote_name: &str,
    create: CreateTaskResult,
    progress: Option<&ToolProgress>,
) -> Result<CallToolResult, ToolError> {
    let peer = client.peer().clone();
    let task_id = create.task.task_id.clone();
    let mut cancel = TaskCancelOnDrop {
        peer: peer.clone(),
        task_id: task_id.clone(),
        finished: false,
    };
    let mut poll_ms = create.task.poll_interval_ms.unwrap_or(1_000).max(10);
    let mut last_status_message: Option<String> = None;

    if let Some(message) = create.task.status_message.as_ref()
        && let Some(sink) = progress
    {
        sink.emit(
            message.clone(),
            Some(json!({
                "taskId": task_id,
                "status": "working",
                "statusMessage": message,
            })),
        );
        last_status_message = Some(message.clone());
    }

    // Cap polls so a stuck server cannot hang the agent forever (~poll_ms * MAX, typically minutes).
    const MAX_POLLS: usize = 10_000;
    for _ in 0..MAX_POLLS {
        tokio::time::sleep(Duration::from_millis(poll_ms)).await;

        let info = peer
            .get_task(GetTaskParams::new(task_id.clone()))
            .await
            .map_err(|e| tool_call_err(server_name, remote_name, e))?;
        let detailed = info.task;

        if let Some(interval) = detailed.task.poll_interval_ms {
            poll_ms = interval.max(10);
        }

        if let Some(message) = detailed.task.status_message.as_ref()
            && last_status_message.as_ref() != Some(message)
        {
            if let Some(sink) = progress {
                let status = match detailed.status() {
                    TaskStatus::Working => "working",
                    TaskStatus::InputRequired => "input_required",
                    TaskStatus::Completed => "completed",
                    TaskStatus::Failed => "failed",
                    TaskStatus::Cancelled => "cancelled",
                    _ => "unknown",
                };
                sink.emit(
                    message.clone(),
                    Some(json!({
                        "taskId": task_id,
                        "status": status,
                        "statusMessage": message,
                    })),
                );
            }
            last_status_message = Some(message.clone());
        }

        match detailed.payload {
            TaskPayload::Working => {}
            TaskPayload::InputRequired { input_requests } => {
                let responses =
                    fulfill_input_requests(&client.service().host, server_name, input_requests)
                        .await?;
                peer.update_task(UpdateTaskParams::new(task_id.clone(), responses))
                    .await
                    .map_err(|e| tool_call_err(server_name, remote_name, e))?;
            }
            TaskPayload::Completed { result } => {
                cancel.finish();
                let call_result: CallToolResult =
                    serde_json::from_value(Value::Object(result)).map_err(|e| {
                        ToolError::Execution(format!(
                            "mcp task `{task_id}` on `{server_name}`/`{remote_name}` returned an invalid CallToolResult: {e}"
                        ))
                    })?;
                return Ok(call_result);
            }
            TaskPayload::Failed { error } => {
                cancel.finish();
                return Err(ToolError::Execution(format!(
                    "mcp task `{task_id}` on `{server_name}`/`{remote_name}` failed: {error:?}"
                )));
            }
            TaskPayload::Cancelled => {
                cancel.finish();
                return Err(ToolError::Execution(format!(
                    "mcp task `{task_id}` on `{server_name}`/`{remote_name}` was cancelled"
                )));
            }
            other => {
                return Err(ToolError::Execution(format!(
                    "mcp task `{task_id}` on `{server_name}`/`{remote_name}` returned unknown payload: {other:?}"
                )));
            }
        }
    }

    let _ = peer
        .cancel_task(CancelTaskParams::new(task_id.clone()))
        .await;
    cancel.finish();
    Err(ToolError::Execution(format!(
        "mcp task `{task_id}` on `{server_name}`/`{remote_name}` did not complete within {MAX_POLLS} polls"
    )))
}

fn tool_output_from_result(
    server_name: &str,
    remote_name: &str,
    result: CallToolResult,
) -> Result<ToolOutput, ToolError> {
    let mut text = String::new();
    let mut images = Vec::new();
    for block in result.content {
        match block {
            ContentBlock::Text(t) => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&t.text);
            }
            ContentBlock::Image(img) => {
                images.push(ImageSource::base64(img.mime_type, img.data));
            }
            // Audio/embedded-resource/resource-link content has no representation in
            // `ToolOutput` today (text + images only, matching every built-in tool). Summarized as
            // text rather than silently dropped, so the model at least knows something came back
            // that it can't fully see — resources/prompts are an explicit v2 scope, not this.
            other => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&format!("[unsupported MCP content block: {other:?}]"));
            }
        }
    }

    if result.is_error == Some(true) {
        return Err(ToolError::Execution(if text.is_empty() {
            format!(
                "mcp server `{server_name}` tool `{remote_name}` reported an error with no message"
            )
        } else {
            text
        }));
    }
    Ok(ToolOutput {
        text,
        images,
        terminate: false,
    })
}

/// Connect to every configured MCP server, returning every tool discovered (already wrapped and ready
/// to [`register`](agent_core::ToolRegistry::register)), a catalog for host RPCs, plus one warning
/// string per server that failed to connect. Fail-soft: a misconfigured or dead server never blocks
/// another configured server, or the agent's own startup — see the module doc comment for why.
///
/// Every server connects *concurrently* (`futures::future::join_all`), not one after another: each
/// connect is independent I/O (a process spawn + handshake, or a network round trip) with zero data
/// dependency on any other server, so connecting sequentially would needlessly add every server's own
/// latency to `run`/`serve` startup instead of paying only the slowest one — a real, user-visible cost
/// for an operator with several servers configured, not a micro-optimization.
pub async fn connect_all(
    configs: &[McpServerConfig],
    idle_reap_after: Duration,
    manifest_dir: Option<&crate::tools::mcp_manifest::ManifestDir>,
) -> (Vec<Arc<dyn Tool>>, McpCatalog, Vec<String>) {
    let jobs: Vec<(McpServerConfig, Dial)> = configs
        .iter()
        .map(|config| (config.clone(), Dial::default()))
        .collect();
    connect_many(&jobs, idle_reap_after, manifest_dir).await
}

/// Connect **one session's own** MCP connectors, named and credentialed by its
/// [session grant](crate::grant), and isolated from every other session on this replica.
///
/// Four differences from [`connect_all`], each a way a host-wide assumption would otherwise leak
/// across tenants:
///
/// - **Headers are pre-resolved.** `secrets` come straight off the grant into [`SecretHeader`] and
///   never through `resolve_config_value` — see that type's doc comment for what a `!command` header
///   would do on a replica.
/// - **No OAuth.** [`oauth_bearer_token`] keys the host's own credential store *by server name*, so a
///   tenant connector named after an operator's login would inherit the operator's token.
/// - **No manifest cache.** That cache is keyed by name + url and shared by the whole replica.
/// - **Egress is checked.** The URL goes through the `web` tool's SSRF layer (and so does every
///   address it resolves to, via the client's resolver), so a connector cannot name the instance
///   metadata service, an EFS mount target, or a peer replica.
///
/// Fail-soft per connector, like `connect_all`: a bad URL or a dead server costs its own tools, not
/// the session. The warnings are for the replica's log — they are not sent to the tenant.
pub async fn connect_granted(
    connectors: &[crate::grant::McpConnector],
    secrets: &BTreeMap<String, Vec<crate::grant::SecretHeader>>,
    egress: &McpEgress,
    host: Arc<McpHost>,
    idle_reap_after: Duration,
) -> (Vec<Arc<dyn Tool>>, McpCatalog, Vec<String>) {
    let mut jobs = Vec::with_capacity(connectors.len());
    let mut warnings = Vec::new();
    for connector in connectors {
        if let Err(e) = egress.check(&connector.url) {
            warnings.push(format!("mcp connector `{}`: {e}", connector.name));
            continue;
        }
        let mut headers = Vec::new();
        for header in secrets.get(&connector.name).into_iter().flatten() {
            match SecretHeader::new(&header.name, header.value.expose()) {
                Ok(h) => headers.push(h),
                // The value is never in the message, and never logged.
                Err(e) => warnings.push(format!("mcp connector `{}`: {e}", connector.name)),
            }
        }
        jobs.push((
            McpServerConfig {
                name: connector.name.clone(),
                transport: McpTransport::Http {
                    url: connector.url.clone(),
                    headers: BTreeMap::new(),
                },
            },
            Dial {
                host: host.clone(),
                http: Some(HttpDial {
                    client: egress.client.clone(),
                    headers,
                }),
            },
        ));
    }
    let (tools, catalog, connect_warnings) = connect_many(&jobs, idle_reap_after, None).await;
    warnings.extend(connect_warnings);
    (tools, catalog, warnings)
}

/// Dial every job concurrently and fold the results — each connect is independent I/O with no data
/// dependency on any other, so connecting one at a time would add every server's latency to startup
/// instead of paying only the slowest.
async fn connect_many(
    jobs: &[(McpServerConfig, Dial)],
    idle_reap_after: Duration,
    manifest_dir: Option<&crate::tools::mcp_manifest::ManifestDir>,
) -> (Vec<Arc<dyn Tool>>, McpCatalog, Vec<String>) {
    let results = futures::future::join_all(
        jobs.iter()
            .map(|(config, dial)| connect_one(config, dial, idle_reap_after, manifest_dir)),
    )
    .await;
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
    let mut catalogs = Vec::new();
    let mut warnings = Vec::new();
    for ((config, _), result) in jobs.iter().zip(results) {
        match result {
            Ok((server_tools, catalog)) => {
                tools.extend(server_tools);
                catalogs.push(catalog);
            }
            Err(e) => {
                tracing::warn!(
                    server = %config.name,
                    error = %e,
                    "failed to connect to MCP server; its tools will not be available"
                );
                warnings.push(format!("mcp server `{}`: {e}", config.name));
            }
        }
    }
    (tools, McpCatalog::new(catalogs), warnings)
}

/// Dial one server and complete the MCP handshake, without listing anything. Split out of
/// [`connect_one`] so [`McpConnection`] can redial the exact same way after a reap — a reconnect must
/// not drift from the original connect.
/// A live client, plus the process group to sweep when it is dropped. HTTP servers have no group —
/// there is no process of ours to reap.
async fn connect_one_client(
    config: &McpServerConfig,
    dial: &Dial,
) -> Result<(McpClient, Option<u32>), String> {
    match &config.transport {
        McpTransport::Stdio { command, args, .. } => {
            connect_stdio(config, dial, command, args).await
        }
        McpTransport::Http { url, .. } => {
            let dialing = connect_http(config, dial, url);
            let client = match dial.http.is_some() {
                // A grant connector is a third-party endpoint on a replica that serves everyone:
                // one that accepts a connection and then says nothing must not hold a session start
                // — or, on a redial after a reap, a tool call — open forever. An operator's own
                // configured server keeps the unbounded wait it always had.
                true => tokio::time::timeout(GRANT_HANDSHAKE_TIMEOUT, dialing)
                    .await
                    .map_err(|_| {
                        format!(
                            "MCP handshake to {} timed out after {}s",
                            redact_url(url),
                            GRANT_HANDSHAKE_TIMEOUT.as_secs()
                        )
                    })?,
                false => dialing.await,
            }?;
            Ok((client, None))
        }
    }
}

/// How long a grant connector gets to finish a handshake. Generous for a round trip to a service
/// that has to be awake anyway, short enough that a dead one fails its own tools rather than the
/// session.
const GRANT_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(20);

/// Rebuild a server's tools from its cached manifest, starting nothing.
///
/// This is the difference between "reaped after 120s" and "never started": a guest that boots and is
/// never asked to browse now spawns no browser server at any point.
fn tools_from_manifest(
    config: &McpServerConfig,
    dial: &Dial,
    manifest: crate::tools::mcp_manifest::ServerManifest,
    idle_reap_after: Duration,
) -> (Vec<Arc<dyn Tool>>, McpServerCatalog) {
    let conn = Arc::new(McpConnection::dormant(
        config.clone(),
        dial.clone(),
        idle_reap_after,
    ));
    register_for_reaping(&conn, idle_reap_after);

    let mut tools: Vec<Arc<dyn Tool>> = manifest
        .tools
        .into_iter()
        .map(|t| {
            Arc::new(McpTool {
                name: registered_name(&config.name, &t.remote_name),
                description: t.description,
                input_schema: t.input_schema,
                remote_name: t.remote_name,
                server_name: config.name.clone(),
                conn: conn.clone(),
            }) as Arc<dyn Tool>
        })
        .collect();

    let mut resource_infos = Vec::with_capacity(manifest.resources.len());
    for resource in manifest.resources {
        let tool_name = registered_resource_name(&config.name, &resource.name);
        resource_infos.push(McpResourceInfo {
            uri: resource.uri.clone(),
            name: resource.name.clone(),
            description: Some(resource.description.clone()).filter(|d| !d.is_empty()),
            tool: tool_name.clone(),
        });
        tools.push(Arc::new(McpResourceTool {
            name: tool_name,
            description: resource.description,
            server_name: config.name.clone(),
            uri: resource.uri,
            conn: conn.clone(),
        }));
    }

    let mut prompt_infos = Vec::with_capacity(manifest.prompts.len());
    for prompt in manifest.prompts {
        let tool_name = registered_prompt_name(&config.name, &prompt.name);
        prompt_infos.push(McpPromptInfo {
            name: prompt.name.clone(),
            description: Some(prompt.description.clone()).filter(|d| !d.is_empty()),
            tool: tool_name.clone(),
        });
        tools.push(Arc::new(McpPromptTool {
            name: tool_name,
            description: prompt.description,
            input_schema: prompt.input_schema,
            server_name: config.name.clone(),
            prompt_name: prompt.name,
            conn: conn.clone(),
        }));
    }

    let catalog = McpServerCatalog {
        name: config.name.clone(),
        conn: Arc::downgrade(&conn),
        resources: resource_infos,
        prompts: prompt_infos,
        protocol_version: None,
    };
    (tools, catalog)
}

async fn connect_one(
    config: &McpServerConfig,
    dial: &Dial,
    idle_reap_after: Duration,
    manifest_dir: Option<&crate::tools::mcp_manifest::ManifestDir>,
) -> Result<(Vec<Arc<dyn Tool>>, McpServerCatalog), String> {
    // Cache hit: advertise from the manifest and start nothing.
    if let Some(manifest) = manifest_dir.and_then(|d| crate::tools::mcp_manifest::load(d, config)) {
        tracing::debug!(
            server = %config.name,
            tools = manifest.tools.len(),
            "advertising MCP tools from the cached manifest; not starting the server"
        );
        return Ok(tools_from_manifest(config, dial, manifest, idle_reap_after));
    }
    let (client, pgid) = connect_one_client(config, dial).await?;
    tools_from_client(config, dial, client, pgid, idle_reap_after, manifest_dir).await
}

/// Spawns `command` as its own process-group leader (`process_group(0)`), the same way
/// `tools::exec::RealRunner` spawns `bash`, so that everything the server forks can be killed with it.
///
/// This used to be deliberately *not* a group leader, on the reasoning that an MCP server is a single
/// long-lived process rather than a shell that can fork off detached descendants. Measurement retired
/// that reasoning: a browser-driving server commonly double-forks its browser, which re-parents to
/// init and survives any kill aimed at the server alone. Killing a `rustwright-mcp` that had opened
/// one page left **16 orphaned Chromium processes holding 322 MB of anonymous memory**, indefinitely —
/// so the reaper would have been freeing 6 MB while stranding 322, and the next call would start a
/// second browser beside the first. (`@playwright/mcp` happens not to do this; "happens not to" is not
/// a property to build a memory budget on.)
///
/// The group is what makes [`sweep_process_group`] able to catch them, and it is why the pid is read
/// here and carried on the connection.
///
/// Dropping the client is still the primary shutdown, not the kill: the MCP stdio contract is that a
/// server watches its stdin for EOF, and the OS closes this end of that pipe however this process
/// exits — a clean return, `std::process::exit`, or a fatal signal. A spec-compliant server (this
/// crate's own `mcp_fixture_stdio_server` test fixture included) sees the EOF and exits on its own,
/// and a well-behaved browser server closes its browser on the way out. The group sweep is the
/// backstop for everything that doesn't.
///
/// `stderr` is left at `TokioChildProcess`'s own default (`Stdio::inherit()`), not captured — a
/// deliberate choice, not an oversight: a server that fails to start or crashes typically explains why
/// on its own stderr, and inheriting it means that reaches the operator's own console (this process's
/// stderr) immediately, the same way a connect failure's `tracing::warn!` does.
async fn connect_stdio(
    config: &McpServerConfig,
    dial: &Dial,
    command: &str,
    args: &[String],
) -> Result<(McpClient, Option<u32>), String> {
    let env = config.resolved_env();
    let child = TokioChildProcess::new(tokio::process::Command::new(command).configure(|cmd| {
        cmd.args(args);
        for (k, v) in &env {
            cmd.env(k, v);
        }
        // Make the server its own process-group leader, so everything it spawns can be taken with it.
        //
        // Dropping the client kills the *server*, and for a server whose children stay in its tree
        // that is enough. It is not enough in general: a browser-driving server commonly double-forks
        // its browser, which re-parents to init and outlives any kill aimed at the server alone.
        // Measured, killing a `rustwright-mcp` that had opened a page left 16 orphaned Chromium
        // processes holding 322 MB of anonymous memory, indefinitely — a reap that reclaims nothing
        // while the next call starts a second browser. A group leader here is what makes
        // [`kill_process_group`] able to catch them.
        cmd.process_group(0);
    }))
    .map_err(|e| format!("failed to spawn `{command}`: {e}"))?;
    // Read before the transport is consumed; this is also the group id, since the child leads it.
    let pgid = child.id();

    let client = McpHandler::new(&config.name, dial.host.clone())
        .serve_with_lifecycle(child, client_lifecycle())
        .await
        .map_err(|e| format!("MCP handshake over stdio failed: {e}"))?;
    Ok((client, pgid))
}

/// The other way a server goes away: the last tool holding the connection is dropped (a `serve`
/// registry rebuild, a finished `run`, process exit). rmcp kills the server itself on drop; this adds
/// the sweep, so that path reclaims as much as a reap does.
impl Drop for McpConnection {
    fn drop(&mut self) {
        // `get_mut` rather than a lock: we hold `&mut self`, so no one else can be holding it.
        if let Some(live) = self.client.get_mut().take() {
            let pgid = live.pgid;
            drop(live);
            sweep_process_group(pgid);
        }
    }
}

/// Sweep a reaped server's process group, in the background.
///
/// Deliberately *after* the client is dropped rather than instead of it: dropping closes the server's
/// stdin, which is the MCP shutdown signal, and a well-behaved server closes its browser on seeing it.
/// This is the backstop for the rest — and it reuses `exec`'s implementation rather than adding a
/// second one, including its `ps`-enumeration pass, because group-signal delivery alone turned out not
/// to be reliable there either.
///
/// What this does *not* cover is the agent being hard-killed: a server in its own group no longer
/// receives the terminal's signals, and nothing then sweeps it. That is no worse than before — an
/// orphaned browser already outlived a killed agent — and fixing it properly needs a supervisor rather
/// than a signal.
fn sweep_process_group(pgid: Option<u32>) {
    let Some(pgid) = pgid else { return };
    // On its own thread: `kill_process_group` blocks (it shells out and sleeps between passes), and
    // this is called from both an async reap and a `Drop`, neither of which may block.
    std::thread::spawn(move || crate::tools::exec::kill_process_group(pgid));
}

/// Dial an HTTP server. Two shapes, decided by `dial.http`:
///
/// - **A grant connector** (`Some`): the caller's client (the daemon's SSRF-checked `mcp_http`) and
///   the grant's own pre-resolved headers. No settings resolution, no OAuth store.
/// - **A configured server** (`None`): its own client, its `settings.json` headers resolved through
///   `resolve_config_value`, and this host's OAuth token for it, if `agent mcp-login` established
///   one.
async fn connect_http(
    config: &McpServerConfig,
    dial: &Dial,
    url: &str,
) -> Result<McpClient, String> {
    let mut custom_headers: HashMap<HeaderName, HeaderValue> = HashMap::new();
    let mut bearer_token = None;
    let client = match &dial.http {
        Some(http) => {
            for header in &http.headers {
                custom_headers.insert(header.name.clone(), header.value.clone());
            }
            http.client.clone()
        }
        None => {
            for (k, v) in config.resolved_headers() {
                let name = match HeaderName::from_bytes(k.as_bytes()) {
                    Ok(name) => name,
                    Err(e) => {
                        tracing::warn!(header = %k, error = %e, "skipping an MCP server header with an invalid name");
                        continue;
                    }
                };
                let value = match HeaderValue::from_str(&v) {
                    Ok(value) => value,
                    Err(e) => {
                        tracing::warn!(header = %k, error = %e, "skipping an MCP server header with an invalid value");
                        continue;
                    }
                };
                custom_headers.insert(name, value);
            }
            // A previously `agent mcp-login`'d server gets its (auto-refreshed, if needed) bearer
            // token attached; a server nobody has logged into (the common case — most MCP servers
            // need no auth at all, or use `headers` above for a static credential) connects exactly
            // as before.
            bearer_token = oauth_bearer_token(&config.name, url).await;
            agent_core::ensure_provider();
            reqwest::Client::new()
        }
    };

    let mut transport_config = StreamableHttpClientTransportConfig::with_uri(url.to_string())
        .custom_headers(custom_headers);
    if let Some(token) = &bearer_token {
        transport_config = transport_config.auth_header(token.clone());
    }
    let transport = StreamableHttpClientTransport::with_client(client, transport_config);

    McpHandler::new(&config.name, dial.host.clone())
        .serve_with_lifecycle(transport, client_lifecycle())
        .await
        .map_err(|e| {
            let hint = match (&dial.http, &bearer_token) {
                // A grant connector has no login to run: its credentials are in the grant.
                (Some(_), _) | (None, Some(_)) => String::new(),
                (None, None) => format!(
                    " (if this server requires login, run `agent mcp-login {}` first)",
                    config.name
                ),
            };
            format!(
                "MCP handshake over streamable-HTTP to {} failed: {e}{hint}",
                redact_url(url)
            )
        })
}

/// The process-wide egress every **grant** MCP connector goes out through, built once by the daemon
/// and shared by every session on the replica.
///
/// One client, not one per session: a `reqwest::Client` is a connection pool, and a replica holding
/// tens of thousands of sessions cannot afford one each. It is deliberately *not* the gateway's
/// shared client — that one is pinned to h2c prior knowledge for one known hop, and a tenant's
/// connector is an arbitrary third-party endpoint.
///
/// Both SSRF layers the `web` tool uses are here, for the same reason they are there: the URL is not
/// this process's to choose. [`SsrfResolver`](crate::tools::web::ssrf::SsrfResolver) validates every
/// address the client actually connects to, and [`Self::check`] rejects a literal internal IP (and a
/// non-`http(s)` scheme) before any DNS happens. Redirects are refused outright: reqwest resolves a
/// redirect target itself, and a redirect to a literal IP would never reach the resolver.
#[derive(Clone)]
pub struct McpEgress {
    client: reqwest::Client,
    policy: Arc<crate::tools::web::ssrf::EgressPolicy>,
}

impl McpEgress {
    pub fn new(policy: crate::tools::web::ssrf::EgressPolicy) -> Result<Self, String> {
        // Before *any* builder call: with reqwest's `rustls-no-provider` a missing process-wide
        // provider panics inside `build()` rather than returning an error.
        agent_core::ensure_provider();
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .pool_idle_timeout(Duration::from_secs(90))
            .redirect(reqwest::redirect::Policy::none())
            .dns_resolver(crate::tools::web::ssrf::SsrfResolver::new(policy.clone()))
            .build()
            .map_err(|e| format!("failed to build the MCP connector client: {e}"))?;
        Ok(Self {
            client,
            policy: Arc::new(policy),
        })
    }

    /// Whether a connector URL may be dialed at all. The message names the URL without its query
    /// string and never mentions the `web` tool's own flags, which have no effect here.
    fn check(&self, url: &str) -> Result<(), String> {
        use crate::tools::web::ssrf::Blocked;
        let parsed: reqwest::Url = url
            .parse()
            .map_err(|e| format!("`{}` is not a valid url: {e}", redact_url(url)))?;
        crate::tools::web::ssrf::validate_url(&parsed, &self.policy).map_err(
            |blocked| match blocked {
                Blocked::Scheme(scheme) => {
                    format!("`{scheme}:` is not an MCP transport; use http or https")
                }
                Blocked::NoHost => format!("`{}` has no host", redact_url(url)),
                Blocked::Address { addr, class, .. } => format!(
                    "refusing to dial `{}`: {addr} is a {class} address",
                    redact_url(url)
                ),
            },
        )
    }
}

/// A currently-valid bearer access token for `server_name`'s MCP OAuth login, if one exists —
/// `agent mcp-login {server_name}` establishes it (see that command's own doc comment); this only
/// ever *reads and refreshes* an already-established one via
/// [`AuthorizationManager::initialize_from_store`]/[`AuthorizationManager::get_access_token`], the
/// same [`ScopedMcpCredentialStore`](crate::mcp_auth_store::ScopedMcpCredentialStore) `mcp-login`
/// wrote to — a refresh (silent, automatic, and re-persisted by `rmcp` itself) is indistinguishable
/// here from a token that never needed refreshing at all.
///
/// Returns `None` — not an error — for every case short of "found a token I could actually attach":
/// no login has ever happened for this server (the overwhelmingly common case, so this stays a single
/// cheap file check via [`McpAuthStore::has_credential`] rather than always paying a metadata-discovery
/// round trip), or a stored credential exists but can no longer be used (refresh token revoked/expired
/// with no way to recover). Either way, [`connect_http`] proceeds with an unauthenticated connect
/// attempt and lets *that* failure (or success, for a server that turns out not to need auth after
/// all) be the actual signal — this function never itself decides a missing/dead credential is fatal.
async fn oauth_bearer_token(server_name: &str, url: &str) -> Option<String> {
    let store = crate::mcp_auth_store::McpAuthStore::open_default();
    if !store.has_credential(server_name) {
        return None;
    }
    let mut manager = match rmcp::transport::auth::AuthorizationManager::new(url).await {
        Ok(manager) => manager,
        Err(e) => {
            tracing::warn!(server = %server_name, error = %e, "failed to set up MCP OAuth for a stored login");
            return None;
        }
    };
    manager.set_credential_store(store.scoped(server_name));
    match manager.initialize_from_store().await {
        Ok(true) => {}
        // `has_credential` above already confirmed a credential exists, so this shouldn't happen in
        // practice — treated the same as "no credential" rather than as an error either way.
        Ok(false) => return None,
        Err(e) => {
            tracing::warn!(server = %server_name, error = %e, "failed to restore a stored MCP OAuth login");
            return None;
        }
    }
    match manager.get_access_token().await {
        Ok(token) => Some(token),
        Err(e) => {
            tracing::warn!(
                server = %server_name,
                error = %e,
                "stored MCP OAuth login could not be refreshed; run `agent mcp-login {server_name}` again"
            );
            None
        }
    }
}

/// One MCP resource, exposed as a zero-arg tool that `resources/read`s a fixed URI.
struct McpResourceTool {
    name: String,
    description: String,
    server_name: String,
    uri: String,
    conn: Arc<McpConnection>,
}

#[async_trait]
impl Tool for McpResourceTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }

    async fn run(&self, _input: Value) -> Result<ToolOutput, ToolError> {
        let client = self.conn.client().await.map_err(|e| {
            ToolError::Execution(format!(
                "mcp server `{}` is not reachable: {e}",
                self.server_name
            ))
        })?;
        let result = client
            .read_resource(ReadResourceRequestParams::new(self.uri.clone()))
            .await
            .map_err(|e| {
                ToolError::Execution(format!(
                    "mcp server `{}` resources/read `{}` failed: {e}",
                    self.server_name, self.uri
                ))
            })?;
        Ok(resource_contents_to_output(&result.contents))
    }
}

/// One MCP prompt, exposed as a tool whose args match the prompt's declared arguments.
struct McpPromptTool {
    name: String,
    description: String,
    input_schema: Value,
    server_name: String,
    prompt_name: String,
    conn: Arc<McpConnection>,
}

#[async_trait]
impl Tool for McpPromptTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Value {
        self.input_schema.clone()
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let arguments = match input {
            Value::Object(map) => Some(map),
            Value::Null => None,
            other => {
                return Err(ToolError::InvalidInput(format!(
                    "expected a JSON object of arguments for `{}`, got: {other}",
                    self.name
                )));
            }
        };
        let client = self.conn.client().await.map_err(|e| {
            ToolError::Execution(format!(
                "mcp server `{}` is not reachable: {e}",
                self.server_name
            ))
        })?;
        let mut params = GetPromptRequestParams::new(self.prompt_name.clone());
        if let Some(arguments) = arguments {
            params = params.with_arguments(arguments);
        }
        let result = client.get_prompt(params).await.map_err(|e| {
            ToolError::Execution(format!(
                "mcp server `{}` prompts/get `{}` failed: {e}",
                self.server_name, self.prompt_name
            ))
        })?;
        let mut text = String::new();
        for msg in result.messages {
            if !text.is_empty() {
                text.push('\n');
            }
            let body = match &msg.content {
                ContentBlock::Text(t) => t.text.clone(),
                other => format!("[{other:?}]"),
            };
            text.push_str(&format!("{:?}: {body}", msg.role));
        }
        if let Some(desc) = result.description {
            if text.is_empty() {
                text = desc;
            } else {
                text = format!("{desc}\n{text}");
            }
        }
        Ok(ToolOutput {
            text,
            images: Vec::new(),
            terminate: false,
        })
    }
}

fn resource_contents_to_output(contents: &[ResourceContents]) -> ToolOutput {
    let mut text = String::new();
    let mut images = Vec::new();
    for block in contents {
        match block {
            ResourceContents::TextResourceContents { text: t, .. } => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(t);
            }
            ResourceContents::BlobResourceContents {
                blob, mime_type, ..
            } => {
                let mime = mime_type.as_deref().unwrap_or("application/octet-stream");
                if mime.starts_with("image/") {
                    images.push(ImageSource::base64(mime.to_string(), blob.clone()));
                } else {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(&format!("[blob {mime}; {} bytes base64]", blob.len()));
                }
            }
            _ => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str("[unsupported MCP resource contents]");
            }
        }
    }
    ToolOutput {
        text,
        images,
        terminate: false,
    }
}

fn prompt_input_schema(arguments: Option<&[rmcp::model::PromptArgument]>) -> Value {
    let mut properties = Map::new();
    let mut required = Vec::new();
    if let Some(args) = arguments {
        for arg in args {
            let mut prop = Map::new();
            prop.insert("type".into(), json!("string"));
            if let Some(desc) = &arg.description {
                prop.insert("description".into(), json!(desc));
            }
            properties.insert(arg.name.clone(), Value::Object(prop));
            if arg.required == Some(true) {
                required.push(arg.name.clone());
            }
        }
    }
    let mut schema = Map::new();
    schema.insert("type".into(), json!("object"));
    schema.insert("properties".into(), Value::Object(properties));
    if !required.is_empty() {
        schema.insert("required".into(), json!(required));
    }
    Value::Object(schema)
}

/// Per-server catalog entry retained after connect — for `get_mcp` / `mcp_complete`.
#[derive(Clone)]
pub struct McpServerCatalog {
    pub name: String,
    /// Weak so a catalog snapshot cannot keep a reaped server's process alive after its tools drop.
    conn: std::sync::Weak<McpConnection>,
    pub resources: Vec<McpResourceInfo>,
    pub prompts: Vec<McpPromptInfo>,
    /// Negotiated peer protocol version, when known.
    pub protocol_version: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct McpResourceInfo {
    pub uri: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub tool: String,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct McpPromptInfo {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub tool: String,
}

/// Every connected server's catalog — completions + diagnostics.
#[derive(Clone, Default)]
pub struct McpCatalog {
    servers: Arc<std::sync::Mutex<Vec<McpServerCatalog>>>,
}

impl McpCatalog {
    pub fn new(servers: Vec<McpServerCatalog>) -> Self {
        Self {
            servers: Arc::new(std::sync::Mutex::new(servers)),
        }
    }

    pub fn snapshot(&self) -> Vec<McpServerCatalog> {
        self.servers
            .lock()
            .ok()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    pub fn filter_enabled(&self, enabled: &McpEnabledSet) -> Vec<McpServerCatalog> {
        self.snapshot()
            .into_iter()
            .filter(|s| enabled.allows(&s.name))
            .collect()
    }

    /// `completion/complete` against a live (or reconnected) server.
    pub async fn complete(
        &self,
        server: &str,
        params: rmcp::model::CompleteRequestParams,
    ) -> Result<rmcp::model::CompleteResult, String> {
        let entry = self
            .snapshot()
            .into_iter()
            .find(|s| s.name == server)
            .ok_or_else(|| format!("unknown MCP server `{server}`"))?;
        let conn = entry
            .conn
            .upgrade()
            .ok_or_else(|| format!("mcp server `{server}` is no longer connected"))?;
        let client = conn.client().await?;
        client
            .complete(params)
            .await
            .map_err(|e| format!("mcp server `{server}` completion/complete failed: {e}"))
    }
}

async fn tools_from_client(
    config: &McpServerConfig,
    dial: &Dial,
    client: McpClient,
    // The server's process group, so a reap can take its children too; `None` for HTTP.
    pgid: Option<u32>,
    idle_reap_after: Duration,
    manifest_dir: Option<&crate::tools::mcp_manifest::ManifestDir>,
) -> Result<(Vec<Arc<dyn Tool>>, McpServerCatalog), String> {
    let remote_tools = client
        .list_all_tools()
        .await
        .map_err(|e| format!("`tools/list` failed: {e}"))?;
    // Fail-soft: a server without resources/prompts capabilities returns an error; treat as empty.
    let remote_resources = match client.list_all_resources().await {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(server = %config.name, error = %e, "resources/list unavailable");
            Vec::new()
        }
    };
    let remote_prompts = match client.list_all_prompts().await {
        Ok(p) => p,
        Err(e) => {
            tracing::debug!(server = %config.name, error = %e, "prompts/list unavailable");
            Vec::new()
        }
    };
    let protocol_version = client
        .peer_info()
        .map(|info| info.protocol_version.to_string());

    let conn = Arc::new(McpConnection::new(
        config.clone(),
        dial.clone(),
        client,
        pgid,
        idle_reap_after,
    ));
    register_for_reaping(&conn, idle_reap_after);

    // Record what this server advertises so the *next* boot can skip starting it entirely. Written
    // after a successful `tools/list`, so a server that failed to enumerate never poisons the cache.
    if let Some(dir) = manifest_dir {
        crate::tools::mcp_manifest::store(
            dir,
            config,
            remote_tools
                .iter()
                .map(|t| crate::tools::mcp_manifest::CachedTool {
                    remote_name: t.name.to_string(),
                    description: t.description.as_deref().unwrap_or_default().to_string(),
                    input_schema: t.schema_as_json_value(),
                })
                .collect(),
            remote_resources
                .iter()
                .map(|r| crate::tools::mcp_manifest::CachedResource {
                    name: r.name.clone(),
                    uri: r.uri.clone(),
                    description: r.description.clone().unwrap_or_else(|| {
                        format!(
                            "MCP resource `{}` ({}) from server `{}`",
                            r.name, r.uri, config.name
                        )
                    }),
                })
                .collect(),
            remote_prompts
                .iter()
                .map(|p| crate::tools::mcp_manifest::CachedPrompt {
                    name: p.name.clone(),
                    description: p.description.clone().unwrap_or_else(|| {
                        format!("MCP prompt `{}` from server `{}`", p.name, config.name)
                    }),
                    input_schema: prompt_input_schema(p.arguments.as_deref()),
                })
                .collect(),
        );
    }

    let mut tools: Vec<Arc<dyn Tool>> = remote_tools
        .into_iter()
        .map(|remote_tool| {
            let description = remote_tool
                .description
                .as_deref()
                .filter(|d| !d.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| {
                    format!(
                        "MCP tool `{}` from server `{}` (no description provided)",
                        remote_tool.name, config.name
                    )
                });
            Arc::new(McpTool {
                name: registered_name(&config.name, &remote_tool.name),
                description,
                input_schema: remote_tool.schema_as_json_value(),
                remote_name: remote_tool.name.into_owned(),
                server_name: config.name.clone(),
                conn: conn.clone(),
            }) as Arc<dyn Tool>
        })
        .collect();

    let mut resource_infos = Vec::new();
    for resource in &remote_resources {
        let tool_name = registered_resource_name(&config.name, &resource.name);
        let description = resource
            .description
            .clone()
            .filter(|d| !d.is_empty())
            .unwrap_or_else(|| {
                format!(
                    "MCP resource `{}` ({}) from server `{}`",
                    resource.name, resource.uri, config.name
                )
            });
        resource_infos.push(McpResourceInfo {
            uri: resource.uri.clone(),
            name: resource.name.clone(),
            description: resource.description.clone(),
            tool: tool_name.clone(),
        });
        tools.push(Arc::new(McpResourceTool {
            name: tool_name,
            description,
            server_name: config.name.clone(),
            uri: resource.uri.clone(),
            conn: conn.clone(),
        }));
    }

    let mut prompt_infos = Vec::new();
    for prompt in &remote_prompts {
        let tool_name = registered_prompt_name(&config.name, &prompt.name);
        let description = prompt
            .description
            .clone()
            .filter(|d| !d.is_empty())
            .unwrap_or_else(|| {
                format!("MCP prompt `{}` from server `{}`", prompt.name, config.name)
            });
        prompt_infos.push(McpPromptInfo {
            name: prompt.name.clone(),
            description: prompt.description.clone(),
            tool: tool_name.clone(),
        });
        tools.push(Arc::new(McpPromptTool {
            name: tool_name,
            description,
            input_schema: prompt_input_schema(prompt.arguments.as_deref()),
            server_name: config.name.clone(),
            prompt_name: prompt.name.clone(),
            conn: conn.clone(),
        }));
    }

    Ok((
        tools,
        McpServerCatalog {
            name: config.name.clone(),
            conn: Arc::downgrade(&conn),
            resources: resource_infos,
            prompts: prompt_infos,
            protocol_version,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_name_uses_the_double_underscore_prefix_convention() {
        assert_eq!(
            registered_name("filesystem", "read_file"),
            "mcp__filesystem__read_file"
        );
    }

    #[test]
    fn registered_name_cannot_collide_with_a_bare_builtin_tool_name() {
        // Every built-in tool name (`read`, `write`, `edit`, `bash`, `ls`, `grep`, `find`, `fork`,
        // `sync`, `logs`) is a bare identifier with no `__` in it — the `mcp__` prefix guarantees no
        // MCP-discovered tool can ever land on one of those keys in the registry, regardless of what a
        // server names its own tool.
        for builtin in ["read", "write", "edit", "bash", "ls", "grep", "find"] {
            assert_ne!(registered_name("server", builtin), builtin);
        }
    }

    #[test]
    fn server_name_from_registered_round_trips() {
        assert_eq!(
            server_name_from_registered("mcp__filesystem__read_file"),
            Some("filesystem")
        );
        assert_eq!(server_name_from_registered("bash"), None);
        assert_eq!(server_name_from_registered("mcp__"), None);
        assert_eq!(server_name_from_registered("mcp__only"), None);
    }

    #[test]
    fn a_secret_header_keeps_its_value_out_of_debug_output() {
        let header = SecretHeader::new("Authorization", "Bearer tenant-token-do-not-log").unwrap();
        let debug = format!("{header:?}");
        assert!(!debug.contains("tenant-token"), "{debug}");
        assert!(debug.contains("authorization"), "{debug}");
        // And through the struct that carries it onto a connection.
        agent_core::ensure_provider();
        let dial = HttpDial {
            client: reqwest::Client::new(),
            headers: vec![header],
        };
        assert!(!format!("{dial:?}").contains("tenant-token"));
    }

    #[test]
    fn a_secret_header_is_taken_literally_and_a_bad_one_never_names_its_value() {
        // `!cmd` and `$VAR` are settings syntax. A grant's header is bytes, not a template.
        let header = SecretHeader::new("X-Token", "!echo pwned").unwrap();
        assert_eq!(header.value.as_bytes(), b"!echo pwned");
        assert_eq!(
            SecretHeader::new("X-Token", "$HOME").unwrap().value,
            "$HOME"
        );
        let err = SecretHeader::new("X-Token", "a\nb").unwrap_err();
        assert!(err.contains("X-Token"), "{err}");
        assert!(!err.contains("a\nb"), "{err}");
        assert!(SecretHeader::new("bad name", "v").is_err());
    }

    #[test]
    fn a_connector_url_loses_its_query_string_in_messages() {
        assert_eq!(
            redact_url("https://mcp.example.com/sse?token=abc123"),
            "https://mcp.example.com/sse?<redacted>"
        );
        assert_eq!(
            redact_url("https://mcp.example.com/sse"),
            "https://mcp.example.com/sse"
        );
    }

    #[test]
    fn grant_connector_egress_refuses_internal_and_non_http_urls() {
        let egress = McpEgress::new(crate::tools::web::ssrf::EgressPolicy::default()).unwrap();
        for url in [
            "http://169.254.169.254/latest/meta-data/",
            "http://10.1.2.3/mcp",
            "http://127.0.0.1:9/mcp",
            "http://[::1]:9/mcp",
        ] {
            let err = egress.check(url).unwrap_err();
            assert!(err.contains("refusing to dial"), "{url}: {err}");
        }
        let err = egress.check("file:///etc/passwd").unwrap_err();
        assert!(err.contains("http"), "{err}");
        // A public endpoint is fine, and its query string never reaches the message.
        assert!(egress.check("https://mcp.example.com/sse?t=secret").is_ok());
        // Opened up (a dev replica), loopback is reachable again.
        let open = McpEgress::new(crate::tools::web::ssrf::EgressPolicy::new(true, &[])).unwrap();
        assert!(open.check("http://127.0.0.1:9/mcp").is_ok());
    }

    /// An elicitation gate that answers with a fixed marker, so a test can tell *which* host
    /// answered a question.
    struct MarkerGate(&'static str);

    #[async_trait]
    impl crate::tools::mcp_host::ElicitationGate for MarkerGate {
        async fn elicit(
            &self,
            _ask: ElicitationAsk,
        ) -> Result<ElicitResult, crate::tools::mcp_host::ElicitationError> {
            Ok(ElicitResult::new(rmcp::model::ElicitationAction::Accept)
                .with_content(json!({ "answered_by": self.0 })))
        }
    }

    fn answered_by(result: &ElicitResult) -> String {
        serde_json::to_value(result)
            .ok()
            .and_then(|v| v.pointer("/content/answered_by").cloned())
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default()
    }

    #[tokio::test]
    async fn an_elicitation_is_answered_by_its_own_connections_host() {
        // Two sessions, each with its own hub and its own gate installed into it — the shape
        // service mode builds. Before this was per connection, the process-wide hub was
        // last-writer-wins and B's gate answered A's server.
        let session_a = Arc::new(McpHost::new());
        let session_b = Arc::new(McpHost::new());
        session_a
            .elicitation
            .install(Arc::new(MarkerGate("session-a")));
        session_b
            .elicitation
            .install(Arc::new(MarkerGate("session-b")));
        // Installed last, and deliberately never the answer: it is what a connection that isn't
        // any one session's would have consulted.
        host().elicitation.install(Arc::new(MarkerGate("process")));

        let a = McpHandler::new("linear", session_a);
        let b = McpHandler::new("linear", session_b);
        let params = ElicitRequestParams::UrlElicitationParams {
            meta: None,
            message: "which session?".into(),
            url: "https://example.com/ask".into(),
            elicitation_id: "e1".into(),
        };
        assert_eq!(answered_by(&a.elicit(params.clone()).await), "session-a");
        assert_eq!(answered_by(&b.elicit(params.clone()).await), "session-b");
        // And a connection built the process-wide way still reaches the process-wide hub.
        let shared = McpHandler::new("linear", host());
        assert_eq!(answered_by(&shared.elicit(params).await), "process");
    }

    #[test]
    fn mcp_enabled_set_defaults_to_all_and_filters() {
        let gate = McpEnabledSet::new();
        assert!(gate.allows("a"));
        assert!(gate.allows("b"));
        gate.set(Some(HashSet::from(["a".into()])));
        assert!(gate.allows("a"));
        assert!(!gate.allows("b"));
        gate.set(Some(HashSet::new()));
        assert!(!gate.allows("a"));
        gate.set(None);
        assert!(gate.allows("a"));
    }
}
