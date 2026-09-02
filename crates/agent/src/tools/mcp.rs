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
//! **Session enablement.** Configured servers stay connected (or lazily dormant) for the process, but
//! which ones are *advertised* is session-scoped via [`McpEnabledSet`] (`serve`'s `set_mcp_enabled`).
//! That is the kit-shaping seam: disable defaults with `--tools`/`--exclude-tools`, then enable only
//! the MCP servers this task needs — without reconnecting or restarting.
//!
//! Out of scope (explicit, not an oversight): MCP *resources* and *prompts*. Adding a brand-new server
//! config mid-process (vs enabling one already configured at startup) is also out of scope.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_core::{ImageSource, Tool, ToolError, ToolOutput, ToolProgress};
use async_trait::async_trait;
use http::{HeaderName, HeaderValue};
use rmcp::model::{
    CallToolRequest, CallToolRequestParams, CallToolResult, ClientRequest, ContentBlock,
    ProgressNotificationParam, ProgressToken, ServerResult,
};
use rmcp::service::{NotificationContext, PeerRequestOptions, RunningService};
use rmcp::transport::streamable_http_client::{
    StreamableHttpClientTransport, StreamableHttpClientTransportConfig,
};
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use rmcp::{ClientHandler, RoleClient, ServiceExt};
use serde_json::{Value, json};

use crate::settings::{McpServerConfig, McpTransport};

/// Routes MCP `notifications/progress` for in-flight `tools/call`s onto the matching
/// [`ToolProgress`] sink. Shared by every tool on one server connection (one handler per
/// [`RunningService`]).
///
/// rmcp always stamps a fresh [`ProgressToken`] onto outbound requests; we register the sink under
/// that token around `await_response` so concurrent calls on the same server cannot cross-wire
/// each other's updates.
#[derive(Clone, Default)]
struct ProgressClient {
    sinks: Arc<std::sync::Mutex<HashMap<ProgressToken, ToolProgress>>>,
}

impl ProgressClient {
    fn register(&self, token: ProgressToken, progress: ToolProgress) {
        if let Ok(mut sinks) = self.sinks.lock() {
            sinks.insert(token, progress);
        }
    }

    fn unregister(&self, token: &ProgressToken) {
        if let Ok(mut sinks) = self.sinks.lock() {
            sinks.remove(token);
        }
    }
}

impl ClientHandler for ProgressClient {
    async fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        // Clone the sink under the lock, emit outside — never hold `std::sync::Mutex` across an
        // await, and never call into user code while the map is locked (re-entrant progress would
        // deadlock).
        let sink = {
            let Ok(sinks) = self.sinks.lock() else {
                return;
            };
            sinks.get(&params.progress_token).cloned()
        };
        let Some(progress) = sink else {
            return;
        };
        let snapshot = format_progress_snapshot(&params);
        let details = progress_details(&params);
        progress.emit(snapshot, Some(details));
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

/// A connected MCP server's live client handle. [`ProgressClient`] receives
/// `notifications/progress`; other server-initiated requests keep trait defaults. Shared (via `Arc`)
/// by every [`McpTool`] the server produced.
type McpClient = RunningService<RoleClient, ProgressClient>;

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
        client: McpClient,
        pgid: Option<u32>,
        idle_after: Duration,
    ) -> Self {
        Self {
            config,
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
    fn dormant(config: McpServerConfig, idle_after: Duration) -> Self {
        Self {
            config,
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
        let (client, pgid) = connect_one_client(&self.config).await?;
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
    /// Shared `tools/call` path. When `progress` is set, registers the sink under rmcp's
    /// auto-assigned `progressToken` for the duration of the call so `notifications/progress`
    /// become `ToolProgress` snapshots.
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

        // Non-progress path: use `call_tool` (same wire shape; slightly less ceremony). Progress path:
        // `send_cancellable_request` so we learn the `progressToken` rmcp stamped onto the request —
        // plain `call_tool` awaits internally and never exposes it, which would make progress
        // unroutable. Matches rmcp's own `test_request_timeout_progress` client pattern.
        let result = if let Some(progress) = progress {
            let handle = client
                .send_cancellable_request(
                    ClientRequest::CallToolRequest(CallToolRequest::new(params)),
                    PeerRequestOptions::no_options(),
                )
                .await
                .map_err(|e| {
                    ToolError::Execution(format!(
                        "mcp server `{}` tool `{}` call failed: {e}",
                        self.server_name, self.remote_name
                    ))
                })?;
            let token = handle.progress_token.clone();
            client.service().register(token.clone(), progress.clone());
            let response = handle.await_response().await;
            client.service().unregister(&token);
            let response = response.map_err(|e| {
                ToolError::Execution(format!(
                    "mcp server `{}` tool `{}` call failed: {e}",
                    self.server_name, self.remote_name
                ))
            })?;
            match response {
                ServerResult::CallToolResult(result) => result,
                _ => {
                    return Err(ToolError::Execution(format!(
                        "mcp server `{}` tool `{}` returned an unexpected response shape",
                        self.server_name, self.remote_name
                    )));
                }
            }
        } else {
            client.call_tool(params).await.map_err(|e| {
                ToolError::Execution(format!(
                    "mcp server `{}` tool `{}` call failed: {e}",
                    self.server_name, self.remote_name
                ))
            })?
        };

        tool_output_from_result(&self.server_name, &self.remote_name, result)
    }
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
/// to [`register`](agent_core::ToolRegistry::register)) plus one warning string per server that failed
/// to connect. Fail-soft: a misconfigured or dead server never blocks another configured server, or the
/// agent's own startup — see the module doc comment for why.
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
) -> (Vec<Arc<dyn Tool>>, Vec<String>) {
    let results = futures::future::join_all(
        configs
            .iter()
            .map(|config| connect_one(config, idle_reap_after, manifest_dir)),
    )
    .await;
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
    let mut warnings = Vec::new();
    for (config, result) in configs.iter().zip(results) {
        match result {
            Ok(server_tools) => tools.extend(server_tools),
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
    (tools, warnings)
}

/// Dial one server and complete the MCP handshake, without listing anything. Split out of
/// [`connect_one`] so [`McpConnection`] can redial the exact same way after a reap — a reconnect must
/// not drift from the original connect.
/// A live client, plus the process group to sweep when it is dropped. HTTP servers have no group —
/// there is no process of ours to reap.
async fn connect_one_client(config: &McpServerConfig) -> Result<(McpClient, Option<u32>), String> {
    match &config.transport {
        McpTransport::Stdio { command, args, .. } => connect_stdio(config, command, args).await,
        McpTransport::Http { url, .. } => connect_http(config, url).await.map(|c| (c, None)),
    }
}

/// Rebuild a server's tools from its cached manifest, starting nothing.
///
/// This is the difference between "reaped after 120s" and "never started": a guest that boots and is
/// never asked to browse now spawns no browser server at any point.
fn tools_from_manifest(
    config: &McpServerConfig,
    manifest: crate::tools::mcp_manifest::ServerManifest,
    idle_reap_after: Duration,
) -> Vec<Arc<dyn Tool>> {
    let conn = Arc::new(McpConnection::dormant(config.clone(), idle_reap_after));
    register_for_reaping(&conn, idle_reap_after);
    manifest
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
        .collect()
}

async fn connect_one(
    config: &McpServerConfig,
    idle_reap_after: Duration,
    manifest_dir: Option<&crate::tools::mcp_manifest::ManifestDir>,
) -> Result<Vec<Arc<dyn Tool>>, String> {
    // Cache hit: advertise from the manifest and start nothing.
    if let Some(manifest) = manifest_dir.and_then(|d| crate::tools::mcp_manifest::load(d, config)) {
        tracing::debug!(
            server = %config.name,
            tools = manifest.tools.len(),
            "advertising MCP tools from the cached manifest; not starting the server"
        );
        return Ok(tools_from_manifest(config, manifest, idle_reap_after));
    }
    let (client, pgid) = connect_one_client(config).await?;
    tools_from_client(config, client, pgid, idle_reap_after, manifest_dir).await
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

    let client = ProgressClient::default()
        .serve(child)
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

async fn connect_http(config: &McpServerConfig, url: &str) -> Result<McpClient, String> {
    let mut custom_headers: HashMap<HeaderName, HeaderValue> = HashMap::new();
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

    // A previously `agent mcp-login`'d server gets its (auto-refreshed, if needed) bearer token
    // attached; a server nobody has logged into (the common case — most MCP servers need no auth at
    // all, or use `headers` above for a static credential) connects exactly as before.
    let bearer_token = oauth_bearer_token(&config.name, url).await;
    let mut transport_config = StreamableHttpClientTransportConfig::with_uri(url.to_string())
        .custom_headers(custom_headers);
    if let Some(token) = &bearer_token {
        transport_config = transport_config.auth_header(token.clone());
    }
    agent_core::ensure_provider();
    let transport =
        StreamableHttpClientTransport::with_client(reqwest::Client::new(), transport_config);

    ProgressClient::default().serve(transport).await.map_err(|e| {
        let hint = if bearer_token.is_none() {
            format!(
                " (if this server requires login, run `agent mcp-login {}` first)",
                config.name
            )
        } else {
            String::new()
        };
        format!("MCP handshake over streamable-HTTP to {url} failed: {e}{hint}")
    })
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

async fn tools_from_client(
    config: &McpServerConfig,
    client: McpClient,
    // The server's process group, so a reap can take its children too; `None` for HTTP.
    pgid: Option<u32>,
    idle_reap_after: Duration,
    manifest_dir: Option<&crate::tools::mcp_manifest::ManifestDir>,
) -> Result<Vec<Arc<dyn Tool>>, String> {
    let remote_tools = client
        .list_all_tools()
        .await
        .map_err(|e| format!("`tools/list` failed: {e}"))?;
    let conn = Arc::new(McpConnection::new(
        config.clone(),
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
        );
    }
    Ok(remote_tools
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
        .collect())
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
