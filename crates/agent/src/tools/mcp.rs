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
//! **Skills (SEP-2640).** A server declaring `io.modelcontextprotocol/skills` has its `skills/list`
//! read at connect and gets one more tool, `mcp__<server>__skill__read`, the verified loading path
//! for its skills; see [`crate::tools::mcp_skills`].
//!
//! Adding a brand-new server config mid-process (vs enabling one already configured at startup) remains
//! out of scope.
//!
//! **Tasks (SEP-2663).** Client advertises `io.modelcontextprotocol/tasks`. When `tools/call` returns
//! `resultType: "task"`, we poll `tasks/get` (honoring the latest `pollIntervalMs`, bounded only by
//! the task's `ttlMs`), fulfill in-task `inputRequests` via `tasks/update` (each key once), surface
//! `statusMessage` as [`ToolProgress`], and best-effort `tasks/cancel` if the tool future is dropped
//! mid-poll or the TTL elapses.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_core::{ImageSource, Tool, ToolError, ToolOutput, ToolProgress};
use async_trait::async_trait;
use http::{HeaderName, HeaderValue};
use rmcp::ServiceError;
use rmcp::model::{
    CallToolRequest, CallToolRequestParams, CallToolResponse, CallToolResult, CancelTaskParams,
    CancelTaskRequest, ClientCapabilities, ClientInfo, ClientRequest, ContentBlock,
    CreateMessageRequestParams, CreateMessageResult, DEFAULT_MRTR_MAX_ROUNDS, ElicitRequestParams,
    ElicitResult, GetPromptRequestParams, GetTaskParams, Implementation, InputRequest,
    InputRequests, InputResponses, ListRootsResult, ProgressNotificationParam, ProgressToken,
    ProtocolVersion, ReadResourceRequestParams, RequestId, ResourceContents, ServerResult,
    TaskPayload, TaskStatus, UpdateTaskParams, UpdateTaskRequest,
};
use rmcp::service::{
    ClientLifecycleMode, ClientServiceExt, InboundStreamOrigin, NotificationContext, Peer,
    PeerRequestOptions, RequestContext, RunningService,
};
use rmcp::transport::ConfigureCommandExt;
use rmcp::transport::streamable_http_client::{
    StreamableHttpClientTransport, StreamableHttpClientTransportConfig,
};
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
    /// Whether this connection advertises the MCP Apps extension (`io.modelcontextprotocol/ui`) on
    /// its handshake. Only ever true for a connection dialed for a session whose client declared it
    /// can render apps — see [`crate::tools::mcp_apps`]. Part of the dial, so a redial after a reap
    /// negotiates exactly what the first dial did.
    apps: bool,
    /// The connection's skills-listing invalidation flag, handed to every handler it dials.
    skills_changed: Arc<std::sync::atomic::AtomicU64>,
}

impl Default for Dial {
    fn default() -> Self {
        Self {
            host: host(),
            http: None,
            apps: false,
            skills_changed: Arc::default(),
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

fn client_info(apps: bool) -> ClientInfo {
    let mut capabilities = ClientCapabilities::builder()
        .enable_elicitation()
        .enable_sampling()
        .enable_roots()
        .enable_tasks()
        .build();
    if apps {
        capabilities
            .extensions
            .get_or_insert_with(Default::default)
            .insert(
                crate::tools::mcp_apps::EXTENSION_ID.to_string(),
                crate::tools::mcp_apps::extension_settings(),
            );
    }
    // SEP-2640 needs nothing from the client side of the handshake (we only issue `skills/*` after
    // seeing the server declare it), but SEP-2133 lets a client say what it understands, and a
    // server may choose what to publish by it.
    capabilities
        .extensions
        .get_or_insert_with(Default::default)
        .insert(
            crate::tools::mcp_skills::SKILLS_EXTENSION_ID.to_string(),
            Default::default(),
        );
    ClientInfo::new(
        capabilities,
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
pub(crate) struct McpHandler {
    server_name: String,
    /// Where this connection's server→client requests (elicitation, sampling) go. Held per
    /// connection rather than read from a process-wide global, so a session that dialed its own
    /// connectors answers their questions itself instead of whichever session installed a gate last.
    host: Arc<McpHost>,
    sinks: Arc<std::sync::Mutex<HashMap<ProgressToken, ToolProgress>>>,
    /// In-flight tool-call progress sinks (LIFO). See [`Self::push_active`].
    active: Arc<std::sync::Mutex<Vec<ToolProgress>>>,
    /// Where this connection's `notifications/events/*` go (MCP Events draft — see
    /// [`crate::tools::mcp_events`]). Empty, and so free, unless a push stream is open.
    events: crate::tools::mcp_events::NotificationRouter,
    /// Advertise MCP Apps on the handshake — see [`Dial::apps`].
    apps: bool,
    /// MCP App view HTML by `ui://` URI, read once per connection and dropped when the server says
    /// its resources changed (`notifications/resources/list_changed`). On the handler, so a redial
    /// (a new process, perhaps new HTML) starts empty.
    views: Arc<std::sync::Mutex<HashMap<String, Arc<rmcp::model::ReadResourceResult>>>>,
    /// The calls in flight on this connection, each with the host of the session that made it.
    /// See [`Self::route`].
    calls: Arc<std::sync::Mutex<Vec<ActiveCall>>>,
    /// Raised by a list-changed notification; the server's SEP-2640 skills listing reads it as
    /// "re-list on next need" (see `mcp_skills::ServerSkills::refresh_if_due`).
    skills_changed: Arc<std::sync::atomic::AtomicU64>,
}

/// One request in flight on a connection, and the session (host) it belongs to.
struct ActiveCall {
    key: u64,
    /// The JSON-RPC id of the outbound request, once sent.
    request: Option<RequestId>,
    host: Arc<McpHost>,
}

impl McpHandler {
    fn new(server_name: impl Into<String>, host: Arc<McpHost>) -> Self {
        Self {
            server_name: server_name.into(),
            host,
            sinks: Arc::new(std::sync::Mutex::new(HashMap::new())),
            active: Arc::new(std::sync::Mutex::new(Vec::new())),
            events: crate::tools::mcp_events::NotificationRouter::default(),
            calls: Arc::new(std::sync::Mutex::new(Vec::new())),
            skills_changed: Arc::default(),
            apps: false,
            views: Arc::default(),
        }
    }

    fn for_dial(server_name: impl Into<String>, dial: &Dial) -> Self {
        Self {
            apps: dial.apps,
            skills_changed: dial.skills_changed.clone(),
            ..Self::new(server_name, dial.host.clone())
        }
    }

    /// Which session a nested server→client request (one the server sends while a call is in
    /// flight) belongs to. Never a guess:
    ///
    /// - Over Streamable HTTP the request arrives on the SSE stream of the POST that raised it
    ///   (`InboundStreamOrigin::OutboundRequest`), so it belongs to that call's session.
    /// - Otherwise (stdio has no such correlation) it is attributable only if every call in flight
    ///   on this connection belongs to one session.
    /// - With no call in flight it goes to this connection's own host.
    /// - With calls from more than one session in flight it is refused: delivered to a possibly
    ///   wrong session, one user would answer another's server, or spend tokens for it.
    fn route(&self, origin: Option<&InboundStreamOrigin>) -> Result<Arc<McpHost>, String> {
        let calls = self
            .calls
            .lock()
            .map_err(|_| "MCP call registry poisoned".to_owned())?;
        if let Some(InboundStreamOrigin::OutboundRequest(id)) = origin
            && let Some(call) = calls.iter().find(|c| c.request.as_ref() == Some(id))
        {
            return Ok(call.host.clone());
        }
        let mut hosts: Vec<&Arc<McpHost>> = Vec::new();
        for call in calls.iter() {
            if !hosts.iter().any(|h| Arc::ptr_eq(h, &call.host)) {
                hosts.push(&call.host);
            }
        }
        match hosts.as_slice() {
            [] => Ok(self.host.clone()),
            [only] => Ok(Arc::clone(only)),
            _ => Err(format!(
                "refused: calls from more than one session are in flight on MCP server `{}` and \
                 this request carries nothing that says which one it belongs to",
                self.server_name
            )),
        }
    }

    /// Register a call in flight for `host` until the returned guard drops; bind its request id
    /// once sent (see [`ActiveCallGuard::bind`]).
    fn track_call(&self, host: Arc<McpHost>) -> ActiveCallGuard {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let key = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Ok(mut calls) = self.calls.lock() {
            calls.push(ActiveCall {
                key,
                request: None,
                host,
            });
        }
        ActiveCallGuard {
            calls: self.calls.clone(),
            key,
        }
    }

    /// Where a server→client elicitation goes (see [`Self::route`]). Split out of the
    /// `ClientHandler` method so the routing is testable without standing up an rmcp
    /// `RequestContext`.
    async fn elicit(
        &self,
        params: ElicitRequestParams,
        origin: Option<&InboundStreamOrigin>,
    ) -> Result<ElicitResult, McpError> {
        let host = self
            .route(origin)
            .map_err(|why| McpError::invalid_request(format!("elicitation {why}"), None))?;
        Ok(host
            .elicitation
            .elicit(ElicitationAsk {
                server: self.server_name.clone(),
                params,
            })
            .await)
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

/// Removes one call from its connection's registry when it ends.
struct ActiveCallGuard {
    calls: Arc<std::sync::Mutex<Vec<ActiveCall>>>,
    key: u64,
}

impl ActiveCallGuard {
    /// Record the call's outbound request id, so a nested request on its stream routes to it.
    fn bind(&self, request: RequestId) {
        if let Ok(mut calls) = self.calls.lock()
            && let Some(call) = calls.iter_mut().find(|c| c.key == self.key)
        {
            call.request = Some(request);
        }
    }
}

impl Drop for ActiveCallGuard {
    fn drop(&mut self) {
        if let Ok(mut calls) = self.calls.lock() {
            calls.retain(|c| c.key != self.key);
        }
    }
}

tokio::task_local! {
    /// The MCP host of the session whose turn is running. `serve` scopes every run (and every
    /// resumed task) in it, so a call answers its server's questions through the session that made
    /// it, even on a connection every session shares.
    static SESSION_HOST: Arc<McpHost>;
}

/// Run `fut` with `host` as the session host for every MCP call made inside it.
pub async fn with_session_host<F: std::future::Future>(host: Arc<McpHost>, fut: F) -> F::Output {
    SESSION_HOST.scope(host, fut).await
}

/// The host that answers a call's questions: the calling session's, else the connection's own.
fn calling_host(connection_host: &Arc<McpHost>) -> Arc<McpHost> {
    SESSION_HOST
        .try_with(Arc::clone)
        .unwrap_or_else(|_| connection_host.clone())
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
        client_info(self.apps)
    }

    // The resource list changed: cached MCP Apps views are stale, and so is the SEP-2640 skills
    // listing. SEP-2640 defines no skills-specific notification (skills are resources, so a server
    // that changes its catalog says so with `notifications/resources/list_changed`); a
    // `notifications/skills/list_changed` (the first-class-primitive design the working group did not
    // adopt) is honoured too, since it can only mean one thing.
    async fn on_resource_list_changed(&self, _context: NotificationContext<RoleClient>) {
        if let Ok(mut views) = self.views.lock() {
            views.clear();
        }
        self.skills_changed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        context: RequestContext<RoleClient>,
    ) -> Result<ElicitResult, McpError> {
        self.elicit(request, context.extensions.get::<InboundStreamOrigin>())
            .await
    }

    async fn create_message(
        &self,
        params: CreateMessageRequestParams,
        context: RequestContext<RoleClient>,
    ) -> Result<CreateMessageResult, McpError> {
        self.route(context.extensions.get::<InboundStreamOrigin>())
            .map_err(|why| McpError::invalid_request(format!("sampling {why}"), None))?
            .sampling
            .create_message(&self.server_name, params)
            .await
    }

    async fn on_custom_notification(
        &self,
        notification: rmcp::model::CustomNotification,
        context: NotificationContext<RoleClient>,
    ) {
        if notification.method == "notifications/skills/list_changed" {
            self.skills_changed
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        // rmcp moves `_meta` off the notification before dispatch. In 3.2.0 it swaps the whole
        // extension map out first, so the metadata lands in `context.extensions` and
        // `context.meta` is left empty; read both, so a fixed rmcp keeps working.
        let subscription_id = context.meta.subscription_id().or_else(|| {
            context
                .extensions
                .get::<rmcp::model::NotificationMetaObject>()
                .and_then(rmcp::model::NotificationMetaObject::subscription_id)
        });
        self.events.route(notification, subscription_id);
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
pub(crate) type McpClient = RunningService<RoleClient, McpHandler>;

/// Session-scoped gate over which configured MCP servers' tools are advertised to the model.
///
/// Configured servers stay connected (or lazily dormant) for the whole process; this only controls
/// which ones enter the [`ToolRegistry`] on each rebuild. `None` means every configured server is
/// enabled (the default, matching prior behavior).
#[derive(Clone, Default)]
pub struct McpEnabledSet {
    inner: Arc<std::sync::Mutex<Option<HashSet<String>>>>,
    /// This session's MCP Apps view, when its client declared it can render apps (`set_mcp_apps`).
    /// Held beside the allow-list because both are the same thing — *which* MCP tools this session
    /// sees — and both reset together on a session switch. See [`crate::tools::mcp_apps`].
    apps: Arc<std::sync::Mutex<Option<Arc<crate::tools::mcp_apps::AppsView>>>>,
    /// The session's SEP-2640 skill state — what it loaded and what its user approved. It rides here
    /// because this set is already the one MCP object that is per session, reset on a session switch
    /// and shared (not copied) with the session's subagents; see [`crate::tools::mcp_skills`].
    skills: Arc<crate::tools::mcp_skills::SkillSession>,
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

    /// This session's skill state (loaded skills, approvals, the approval gate).
    pub fn skill_session(&self) -> &Arc<crate::tools::mcp_skills::SkillSession> {
        &self.skills
    }

    /// A session switch: back to every server enabled, and forget what the outgoing session loaded
    /// and approved — an incoming session inherits neither.
    pub fn reset_session(&self) {
        self.set(None);
        self.skills.reset();
    }

    /// Whether tools from `server` should be advertised.
    pub fn allows(&self, server: &str) -> bool {
        match self.snapshot() {
            None => true,
            Some(set) => set.contains(server),
        }
    }

    /// Install (or with `None`, clear) this session's MCP Apps view.
    pub fn set_apps(&self, view: Option<Arc<crate::tools::mcp_apps::AppsView>>) {
        if let Ok(mut guard) = self.apps.lock() {
            *guard = view;
        }
    }

    /// This session's MCP Apps view, if its client declared it can render apps.
    pub fn apps(&self) -> Option<Arc<crate::tools::mcp_apps::AppsView>> {
        self.apps.lock().ok().and_then(|g| g.clone())
    }
}

/// The prefix every MCP-discovered tool's registered name carries — `mcp__<server>__<tool>` — so it
/// can never collide with a built-in tool. Matches the convention Claude Code itself uses for the
/// identical problem.
pub(crate) fn registered_name(server: &str, remote_tool: &str) -> String {
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

/// Keep only MCP tools whose server is currently enabled — each skill-loading tool rebound to this
/// session's skill state, so what one session loaded never stands for another.
///
/// A session that declared it renders MCP Apps sees its apps view's tools instead of `tools` — the
/// same servers, dialed with the extension advertised (see [`crate::tools::mcp_apps`]).
pub fn filter_by_enabled(tools: &[Arc<dyn Tool>], enabled: &McpEnabledSet) -> Vec<Arc<dyn Tool>> {
    filter_by_enabled_as(tools, enabled, &crate::approval::ApprovalOrigin::Main)
}

/// [`filter_by_enabled`] for the agent `origin` — a subagent's registry — so a skill load it makes
/// asks the user in that subagent's name.
pub fn filter_by_enabled_as(
    tools: &[Arc<dyn Tool>],
    enabled: &McpEnabledSet,
    origin: &crate::approval::ApprovalOrigin,
) -> Vec<Arc<dyn Tool>> {
    // A declared session sees the apps flavor's tools for every server whose apps connection is up,
    // and the plain tools for every other one — a server whose apps dial failed (or has not landed
    // yet) keeps working as a text tool rather than vanishing.
    let merged;
    let tools = match enabled.apps() {
        Some(view) => {
            merged = view.tools_over(tools);
            &merged[..]
        }
        None => tools,
    };
    tools
        .iter()
        .filter(|t| match server_name_from_registered(t.name()) {
            Some(server) => enabled.allows(server),
            None => true,
        })
        .map(|t| {
            enabled
                .skills
                .bind_as(t, origin)
                .unwrap_or_else(|| t.clone())
        })
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
/// Dropping the client is what ends the child: rmcp drops the stdio transport, and
/// `mcp_stdio::retire` closes its stdin, gives it its grace, then kills and reaps what is left.
/// A connected server: the client, and the process group to sweep when it goes away.
///
/// The group is kept beside the client rather than on the connection because it belongs to *this*
/// process — a reconnect after a reap starts a new one, and sweeping the old group id then would
/// either do nothing or, if the id had been recycled, kill something unrelated.
struct Live {
    client: Arc<McpClient>,
    /// `None` for HTTP transports: there is no process of ours to reap.
    proc: ServerProc,
}

/// A stdio server's process, to retire when its connection goes away; `None` for HTTP.
type ServerProc = Option<Arc<crate::tools::mcp_stdio::ServerProcess>>;

pub(crate) struct McpConnection {
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
        proc: ServerProc,
        idle_after: Duration,
    ) -> Self {
        Self {
            config,
            dial,
            client: tokio::sync::Mutex::new(Some(Live {
                client: Arc::new(client),
                proc,
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
    pub(crate) async fn client(&self) -> Result<Arc<McpClient>, String> {
        self.last_used
            .store(now_secs(), std::sync::atomic::Ordering::Relaxed);
        let mut guard = self.client.lock().await;
        if let Some(live) = guard.as_ref() {
            return Ok(live.client.clone());
        }
        let (client, proc) = connect_one_client(&self.config, &self.dial).await?;
        let client = Arc::new(client);
        *guard = Some(Live {
            client: client.clone(),
            proc,
        });
        Ok(client)
    }

    /// Forget `stale` if it is still the cached client, so the next [`Self::client`] redials. A
    /// client someone else already replaced is left alone: two calls noticing the same dead
    /// connection must produce one redial, not drop each other's fresh client.
    async fn invalidate(&self, stale: &Arc<McpClient>) {
        let mut guard = self.client.lock().await;
        if let Some(live) = guard.as_ref()
            && Arc::ptr_eq(&live.client, stale)
        {
            let proc = live.proc.clone();
            *guard = None;
            if let Some(proc) = &proc {
                crate::tools::mcp_stdio::retire(proc);
            }
        }
    }

    /// The client only if one is live right now — never dials. For work that is worth doing while a
    /// server is up but not worth starting one for (re-listing skills), and that must not count as
    /// use, so it leaves `last_used` alone.
    pub(crate) async fn live_client(&self) -> Option<Arc<McpClient>> {
        self.client
            .lock()
            .await
            .as_ref()
            .map(|live| live.client.clone())
    }

    /// The server's configuration — what its manifest is cached under.
    pub(crate) fn config(&self) -> &McpServerConfig {
        &self.config
    }

    /// The flag a list-changed notification raises on this connection, whichever dial received it.
    pub(crate) fn skills_changed(&self) -> Arc<std::sync::atomic::AtomicU64> {
        self.dial.skills_changed.clone()
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
                let proc = live.proc.clone();
                let pgid = proc.as_ref().and_then(|p| p.pgid());
                // Drops the client; then the server's stdin closes, it gets
                // `mcp_stdio::SHUTDOWN_GRACE` to exit, and its group is swept — taking anything it
                // forked away from itself, which a kill aimed at the server alone leaves running.
                *guard = None;
                if let Some(proc) = &proc {
                    crate::tools::mcp_stdio::retire(proc);
                }
                tracing::debug!(
                    server = %self.config.name,
                    pgid,
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
        let result = call_tool_raw(
            &self.conn,
            &self.server_name,
            &self.remote_name,
            arguments,
            progress,
        )
        .await?;
        tool_output_from_result(&self.server_name, &self.remote_name, result)
    }
}

/// One `tools/call` on `conn`, returning the server's whole result — `structuredContent` and
/// `_meta` included, which the model never sees but an MCP App view does.
async fn call_tool_raw(
    conn: &Arc<McpConnection>,
    server_name: &str,
    remote_name: &str,
    arguments: Option<Map<String, Value>>,
    progress: Option<&ToolProgress>,
) -> Result<CallToolResult, ToolError> {
    let mut params = CallToolRequestParams::new(remote_name.to_string());
    if let Some(arguments) = arguments {
        params = params.with_arguments(arguments);
    }
    let client = conn.client().await.map_err(|e| {
        ToolError::Execution(format!("mcp server `{server_name}` is not reachable: {e}"))
    })?;
    // The questions this call raises go to the session that made it (see [`calling_host`]) —
    // tracked per call, so a nested request is attributed to that session or refused.
    let host = calling_host(&client.service().host);
    let _active = progress.map(|p| client.service().push_active(p.clone()));
    drive_tool_call(
        conn,
        client.clone(),
        &host,
        server_name,
        remote_name,
        params,
        progress,
    )
    .await
}

/// The session host the current task runs under (`with_session_host`), if any — for work an MCP
/// App spawns off its call's task (a view read) that must still answer to that session.
pub(crate) fn current_session_host() -> Option<Arc<McpHost>> {
    SESSION_HOST.try_with(Arc::clone).ok()
}

/// A strong handle on one server's connection, for the MCP Apps host bridge
/// ([`crate::tools::mcp_apps`]): the model-side tools are not the only thing that calls a server
/// once a view can call it back.
#[derive(Clone)]
pub(crate) struct McpServerHandle {
    name: String,
    conn: Arc<McpConnection>,
}

impl McpServerHandle {
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) async fn call_tool(
        &self,
        remote_name: &str,
        arguments: Option<Map<String, Value>>,
        progress: Option<&ToolProgress>,
    ) -> Result<CallToolResult, ToolError> {
        call_tool_raw(&self.conn, &self.name, remote_name, arguments, progress).await
    }

    /// An MCP App view's `resources/read`, served from this connection's cache when it can be — the
    /// spec lets a host cache view templates, and a view is re-read on every call of its tool
    /// otherwise. Invalidated by the server's `resources/list_changed` and by any redial.
    pub(crate) async fn read_view(
        &self,
        uri: &str,
    ) -> Result<Arc<rmcp::model::ReadResourceResult>, String> {
        let client = self.conn.client().await?;
        let views = client.service().views.clone();
        if let Some(hit) = views.lock().ok().and_then(|v| v.get(uri).cloned()) {
            return Ok(hit);
        }
        // Tracked like every call: a nested request the server raises meanwhile is attributed to
        // the calling session (or refused when that is ambiguous).
        let _call = client
            .service()
            .track_call(calling_host(&client.service().host));
        let read = Arc::new(
            client
                .read_resource(ReadResourceRequestParams::new(uri.to_string()))
                .await
                .map_err(|e| {
                    format!(
                        "mcp server `{}` resources/read `{uri}` failed: {e}",
                        self.name
                    )
                })?,
        );
        // Never cache a view this host would refuse anyway: one oversized read must not pin its
        // bytes for the connection's life.
        let size: usize = read
            .contents
            .iter()
            .map(|c| match c {
                ResourceContents::TextResourceContents { text, .. } => text.len(),
                ResourceContents::BlobResourceContents { blob, .. } => blob.len(),
                #[allow(unreachable_patterns)]
                _ => 0,
            })
            .sum();
        if size <= crate::tools::mcp_apps::MAX_VIEW_BYTES
            && let Ok(mut v) = views.lock()
        {
            v.insert(uri.to_string(), read.clone());
        }
        Ok(read)
    }

    pub(crate) async fn read_resource(
        &self,
        uri: &str,
    ) -> Result<rmcp::model::ReadResourceResult, String> {
        let client = self.conn.client().await?;
        let _call = client
            .service()
            .track_call(calling_host(&client.service().host));
        client
            .read_resource(ReadResourceRequestParams::new(uri.to_string()))
            .await
            .map_err(|e| {
                format!(
                    "mcp server `{}` resources/read `{uri}` failed: {e}",
                    self.name
                )
            })
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
    conn: &Arc<McpConnection>,
    client: Arc<McpClient>,
    host: &McpHost,
    server_name: &str,
    remote_name: &str,
    mut params: CallToolRequestParams,
    progress: Option<&ToolProgress>,
) -> Result<CallToolResult, ToolError> {
    for _round in 0..DEFAULT_MRTR_MAX_ROUNDS {
        let host_arc = calling_host(&client.service().host);
        match call_tool_tracked(&client, params.clone(), host_arc)
            .await
            .map_err(|e| tool_call_err(server_name, remote_name, e))?
        {
            CallToolResponse::Complete(result) => return Ok(result),
            CallToolResponse::InputRequired(required) => {
                let responses = fulfill_input_requests(
                    host,
                    server_name,
                    required.input_requests.unwrap_or_default(),
                )
                .await?;
                params.input_responses = (!responses.is_empty()).then_some(responses);
                params.request_state = required.request_state;
            }
            CallToolResponse::Task(create) => {
                let task = &create.task;
                let record = McpTaskRecord {
                    server: server_name.to_owned(),
                    tool: remote_name.to_owned(),
                    task_id: task.task_id.clone(),
                    created_at_ms: unix_ms(),
                    ttl_ms: task.ttl_ms,
                    answered: Vec::new(),
                };
                let seed = TaskSeed {
                    poll_interval_ms: task.poll_interval_ms,
                    status_message: task.status_message.clone(),
                    cancel_on_drop: true,
                    fresh: true,
                };
                return await_task(conn, client, host, record, seed, progress).await;
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

/// One `tools/call`, registered as in flight (with its request id) for exactly as long as the
/// request is outstanding, so a nested request the server sends meanwhile can be attributed to the
/// calling session (see [`McpHandler::route`]). Once the response (a result, an MRTR round, or a
/// task handle) is back, nothing nested can belong to it any more.
async fn call_tool_tracked(
    client: &McpClient,
    params: CallToolRequestParams,
    host: Arc<McpHost>,
) -> Result<CallToolResponse, ServiceError> {
    let call = client.service().track_call(host);
    let handle = client
        .peer()
        .send_request_with_option(
            ClientRequest::CallToolRequest(CallToolRequest::new(params)),
            PeerRequestOptions::no_options(),
        )
        .await?;
    call.bind(handle.id.clone());
    match handle.await_response().await? {
        ServerResult::CallToolResult(result) => Ok(CallToolResponse::Complete(result)),
        ServerResult::InputRequiredResult(result) => Ok(CallToolResponse::InputRequired(result)),
        ServerResult::CreateTaskResult(result) => Ok(CallToolResponse::Task(result)),
        _ => Err(ServiceError::UnexpectedResponse),
    }
}

/// Any other request a session makes of a server (the skills extension's `skills/*` and
/// `resources/read`), tracked like [`call_tool_tracked`] under the calling session's host (the
/// task-local [`with_session_host`] scope, else the connection's own), so a nested request the server
/// raises meanwhile is attributed to that session or refused, never guessed.
pub(crate) async fn request_tracked(
    client: &McpClient,
    request: ClientRequest,
) -> Result<ServerResult, ServiceError> {
    let call = client
        .service()
        .track_call(calling_host(&client.service().host));
    let handle = client
        .peer()
        .send_request_with_option(request, PeerRequestOptions::no_options())
        .await?;
    call.bind(handle.id.clone());
    handle.await_response().await
}

/// The `tool_progress` `details` key carrying a task's current [`McpTaskRecord`].
pub const MCP_TASK_DETAILS_KEY: &str = "mcpTask";

/// The session custom-entry kind `serve` journals an [`McpTaskRecord`] under.
pub const MCP_TASK_ENTRY_KIND: &str = "mcp_task";

/// A durable handle to one in-flight SEP-2663 task: enough to poll it again from a fresh process.
///
/// The spec asks clients to persist task ids so polling survives a crash or restart. The tool emits
/// this record in a `tool_progress` `details` entry ([`MCP_TASK_DETAILS_KEY`]) when the task is
/// created, and again whenever its durable state changes (an `inputRequests` key answered, the
/// `ttlMs` moved). `serve` journals each emission with the session (see `crate::mcp_resume`); the
/// latest one for a task wins.
///
/// The `taskId` may be a bearer token for the server's stored state, so the journal is as private
/// as the session file that holds it, and the HTML export leaves it out.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct McpTaskRecord {
    pub server: String,
    /// The server's own tool name (not the registered `mcp__…` name).
    pub tool: String,
    pub task_id: String,
    /// Local wall clock when the task was created, so the `ttlMs` backstop still counts from
    /// creation after a restart.
    pub created_at_ms: u64,
    /// The task's latest `ttlMs`, so a resumed task keeps its backstop even if the server can no
    /// longer be reached to say it again.
    #[serde(default)]
    pub ttl_ms: Option<u64>,
    /// `inputRequests` keys already answered, so a resume never asks the user twice.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub answered: Vec<String>,
}

pub fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// Fulfill SEP-2322 / in-task `inputRequests` through the hubs of the host that owns this call
/// (see [`calling_host`]).
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
                    .create_message(server_name, req.params)
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

/// Send `tasks/update` and take any successful result as the acknowledgement.
///
/// The spec's ack is an empty result, and every MCP result may carry `_meta`. rmcp's
/// `Peer::update_task` decodes the reply through its untagged `ServerResult`, where
/// `{"_meta":{…},"resultType":"complete"}` matches `CallToolResult` first (its `content` defaults
/// to empty) and the ack is rejected as an unexpected response: found against the SEP's reference
/// server (mcpkit), whose acks carry `serverInfo` in `_meta`. A JSON-RPC error still comes back as
/// an error; only the shape of a success is not second-guessed.
async fn update_task_acked(
    peer: &Peer<RoleClient>,
    params: UpdateTaskParams,
) -> Result<(), ServiceError> {
    peer.send_request(ClientRequest::UpdateTaskRequest(UpdateTaskRequest::new(
        params,
    )))
    .await
    .map(drop)
}

/// `tasks/cancel`, with the same any-success-is-the-ack reading as [`update_task_acked`].
async fn cancel_task_acked(
    peer: &Peer<RoleClient>,
    params: CancelTaskParams,
) -> Result<(), ServiceError> {
    peer.send_request(ClientRequest::CancelTaskRequest(CancelTaskRequest::new(
        params,
    )))
    .await
    .map(drop)
}

/// Best-effort `tasks/cancel` when the tool future is aborted mid-poll.
///
/// Holds the connection, not a peer: a call aborted mid-backoff would otherwise cancel through the
/// client that just died. At drop it uses whatever client the connection has *now* (one redialed
/// since the loss), redialing for it if there is none: a server that keeps its tasks across its own
/// restart still gets the cancel.
///
/// Disarmed (`armed: false`) for a resume after a restart: the user aborting the prompt that waited
/// on it did not ask to cancel the task, which the next prompt will wait on again.
struct TaskCancelOnDrop {
    conn: Arc<McpConnection>,
    task_id: String,
    armed: bool,
}

impl TaskCancelOnDrop {
    fn finish(&mut self) {
        self.armed = false;
    }
}

impl Drop for TaskCancelOnDrop {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let conn = self.conn.clone();
        let task_id = self.task_id.clone();
        tokio::spawn(async move {
            let client = match conn.live_client().await {
                Some(client) => Some(client),
                None => conn.client().await.ok(),
            };
            if let Some(client) = client {
                let _ = cancel_task_acked(client.peer(), CancelTaskParams::new(task_id)).await;
            }
        });
    }
}

/// How polling starts: from a fresh `CreateTaskResult`, or (resume) from a journaled record.
struct TaskSeed {
    poll_interval_ms: Option<u64>,
    status_message: Option<String>,
    /// Whether dropping the poll cancels the task (see [`TaskCancelOnDrop`]).
    cancel_on_drop: bool,
    /// A task just created (journal its record) rather than one resumed from the journal, whose
    /// record is already there: a restart must not append another per pending task.
    fresh: bool,
}

/// Consecutive connection losses one task survives; the next one fails the call.
const MAX_TASK_RECONNECTS: u32 = 8;

/// The longest single wait between polls, whatever `pollIntervalMs` says, so a TTL or an abort is
/// noticed without sleeping through it.
const MAX_POLL_SLEEP: Duration = Duration::from_secs(60);

/// What a failed request says about the connection.
enum Loss {
    /// The request could not be delivered, but the client is alive (an HTTP POST that failed):
    /// back off and send again on the same client.
    Retry,
    /// The client is gone (transport closed, e.g. a stdio server that exited): redial.
    Redial,
}

/// A failed request that means the connection is in trouble, not that the server answered. The
/// task lives on the server, so the right response is to recover and poll the same `taskId`.
fn connection_loss(e: &ServiceError) -> Option<Loss> {
    match e {
        ServiceError::TransportSend(_) => Some(Loss::Retry),
        ServiceError::TransportClosed => Some(Loss::Redial),
        _ => None,
    }
}

/// Back off, then (for [`Loss::Redial`]) drop the dead client and dial again. A failed dial keeps
/// the stale client: the next request on it fails as a loss too, which counts toward
/// [`MAX_TASK_RECONNECTS`].
async fn recover(
    conn: &McpConnection,
    client: Arc<McpClient>,
    loss: Loss,
    losses: u32,
) -> Arc<McpClient> {
    let backoff = Duration::from_millis(200u64 << losses.saturating_sub(1).min(5));
    tokio::time::sleep(backoff.min(Duration::from_secs(5))).await;
    if let Loss::Retry = loss {
        return client;
    }
    conn.invalidate(&client).await;
    match conn.client().await {
        Ok(fresh) => fresh,
        Err(e) => {
            tracing::debug!(server = %conn.config.name, error = %e, "mcp task redial failed");
            client
        }
    }
}

/// Emit the task's current durable record, for `serve` to journal (see [`McpTaskRecord`]).
fn journal(progress: Option<&ToolProgress>, record: &McpTaskRecord, snapshot: &str) {
    if let Some(sink) = progress {
        sink.emit(
            snapshot.to_owned(),
            Some(json!({ "taskId": record.task_id, MCP_TASK_DETAILS_KEY: record })),
        );
    }
}

/// Poll `tasks/get` until terminal; fulfill in-task input via `tasks/update`.
///
/// No poll-count cap: the spec says to keep polling until a terminal status or `tasks/cancel`, and a
/// cap measured in polls is a wall-clock limit that silently depends on the server's
/// `pollIntervalMs`. The only backstop is the task's own `ttlMs` (latest value, counted from
/// creation), which the spec lets a client treat as the point the task is no longer usable; a
/// `null` TTL polls until the turn is aborted. A single wait is capped by the TTL's remainder and
/// [`MAX_POLL_SLEEP`].
///
/// An `inputRequests` key is answered once. `tasks/update` is acknowledged eventually-consistently,
/// so the next poll can still list a key we already answered; re-asking would show the user the
/// same elicitation twice and send a duplicate response. Answered keys are journaled with the
/// record, so a resume after a restart does not ask again either.
///
/// A lost connection is not the task's failure: the task lives on the server. A request that could
/// not be delivered is re-sent on the same client after a backoff; a dead client is dropped and the
/// server redialed. Either way the same `taskId` is polled again, and a pending `tasks/update` is
/// re-sent rather than re-asked. The call survives [`MAX_TASK_RECONNECTS`] consecutive losses and
/// fails on the next. A stdio server that died took its tasks with it: the redialed process
/// answers `-32602`, and that ends the call.
async fn await_task(
    conn: &Arc<McpConnection>,
    mut client: Arc<McpClient>,
    host: &McpHost,
    mut record: McpTaskRecord,
    seed: TaskSeed,
    progress: Option<&ToolProgress>,
) -> Result<CallToolResult, ToolError> {
    let task_id = record.task_id.clone();
    let server_name = record.server.clone();
    let remote_name = record.tool.clone();
    let (server_name, remote_name) = (server_name.as_str(), remote_name.as_str());
    let mut cancel = TaskCancelOnDrop {
        conn: conn.clone(),
        task_id: task_id.clone(),
        armed: seed.cancel_on_drop,
    };
    let created = Instant::now()
        .checked_sub(Duration::from_millis(
            unix_ms().saturating_sub(record.created_at_ms),
        ))
        .unwrap_or_else(Instant::now);
    let mut poll_ms = seed.poll_interval_ms.unwrap_or(1_000).max(10);
    let mut last_status_message = seed.status_message;
    let mut answered: HashSet<String> = record.answered.iter().cloned().collect();
    let mut pending_update: Option<(Vec<String>, InputResponses)> = None;
    let mut losses: u32 = 0;
    let lost = |losses: u32, e: &ServiceError| {
        ToolError::Execution(format!(
            "mcp task `{task_id}` on `{server_name}`/`{remote_name}`: connection lost {losses} times in a row, giving up: {e}"
        ))
    };

    if seed.fresh {
        journal(
            progress,
            &record,
            last_status_message.as_deref().unwrap_or("task created"),
        );
    }

    loop {
        if let Some((keys, responses)) = pending_update.take() {
            match update_task_acked(
                client.peer(),
                UpdateTaskParams::new(task_id.clone(), responses.clone()),
            )
            .await
            {
                Ok(()) => {
                    answered.extend(keys);
                    losses = 0;
                    record.answered = answered.iter().cloned().collect();
                    record.answered.sort();
                    journal(progress, &record, "input delivered");
                }
                Err(e) => match connection_loss(&e) {
                    Some(loss) => {
                        losses += 1;
                        if losses > MAX_TASK_RECONNECTS {
                            return Err(lost(losses, &e));
                        }
                        pending_update = Some((keys, responses));
                        client = recover(conn, client, loss, losses).await;
                        continue;
                    }
                    None => return Err(tool_call_err(server_name, remote_name, e)),
                },
            }
        }

        let mut wait = Duration::from_millis(poll_ms).min(MAX_POLL_SLEEP);
        if let Some(ttl) = record.ttl_ms {
            let ttl = Duration::from_millis(ttl);
            let elapsed = created.elapsed();
            if elapsed >= ttl {
                // Not `finish()`ed: the drop guard sends a best-effort `tasks/cancel` (unless this
                // is a resume, which never cancels).
                return Err(ToolError::Execution(format!(
                    "mcp task `{task_id}` on `{server_name}`/`{remote_name}` did not finish within its ttlMs ({} ms)",
                    ttl.as_millis()
                )));
            }
            wait = wait.min(ttl - elapsed);
        }
        tokio::time::sleep(wait).await;

        let info = match client
            .peer()
            .get_task(GetTaskParams::new(task_id.clone()))
            .await
        {
            Ok(info) => {
                losses = 0;
                info
            }
            Err(e) => match connection_loss(&e) {
                Some(loss) => {
                    losses += 1;
                    if losses > MAX_TASK_RECONNECTS {
                        return Err(lost(losses, &e));
                    }
                    client = recover(conn, client, loss, losses).await;
                    continue;
                }
                None => return Err(tool_call_err(server_name, remote_name, e)),
            },
        };
        let detailed = info.task;

        if let Some(interval) = detailed.task.poll_interval_ms {
            poll_ms = interval.max(10);
        }
        if detailed.task.ttl_ms != record.ttl_ms {
            record.ttl_ms = detailed.task.ttl_ms;
            journal(
                progress,
                &record,
                last_status_message.as_deref().unwrap_or("ttl changed"),
            );
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
            TaskPayload::InputRequired { mut input_requests } => {
                input_requests.retain(|key, _| !answered.contains(key));
                if input_requests.is_empty() {
                    continue;
                }
                let keys: Vec<String> = input_requests.keys().cloned().collect();
                let responses = fulfill_input_requests(host, server_name, input_requests).await?;
                pending_update = Some((keys, responses));
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
                    "mcp task `{task_id}` on `{server_name}`/`{remote_name}` failed: {}",
                    Value::Object(error)
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
}

impl McpCatalog {
    /// Resume polling a journaled task after a restart (see [`McpTaskRecord`]) through this
    /// catalog's live (or redialed) connection to `record.server`, to its terminal result. Never
    /// cancels the task when dropped. A server that no longer knows the task answers `-32602`,
    /// and that is the call's answer. `host` answers any in-task input; `progress` receives the
    /// task's status and record updates.
    pub async fn resume_task(
        &self,
        record: McpTaskRecord,
        host: &McpHost,
        progress: Option<&ToolProgress>,
    ) -> Result<ToolOutput, ToolError> {
        let server = record.server.clone();
        let tool = record.tool.clone();
        // A TTL that ran out while the process was down needs no server (nor its configuration)
        // to say so.
        if let Some(ttl) = record.ttl_ms
            && unix_ms().saturating_sub(record.created_at_ms) >= ttl
        {
            return Err(ToolError::Execution(format!(
                "mcp task `{}` on `{server}`/`{tool}` did not finish within its ttlMs ({ttl} ms)",
                record.task_id
            )));
        }
        let entry = self
            .snapshot()
            .into_iter()
            .find(|s| s.name == server)
            .ok_or_else(|| {
                ToolError::Execution(format!("mcp server `{server}` is not configured any more"))
            })?;
        let conn = entry.conn.upgrade().ok_or_else(|| {
            ToolError::Execution(format!("mcp server `{server}` is no longer connected"))
        })?;
        let seed = TaskSeed {
            // Unknown until the first poll answers; ask promptly, then follow the server.
            poll_interval_ms: Some(10),
            status_message: None,
            cancel_on_drop: false,
            fresh: false,
        };
        let client = conn.client().await.map_err(|e| {
            ToolError::Execution(format!("mcp server `{server}` is not reachable: {e}"))
        })?;
        let result = await_task(&conn, client, host, record, seed, progress).await?;
        tool_output_from_result(&server, &tool, result)
    }
}

pub(crate) fn tool_output_from_result(
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

/// The same servers as [`connect_all`], dialed **with MCP Apps advertised** — lazily, on the first
/// `get`, which is the first session whose client declares it can render apps. A separate set of
/// connections rather than a flag on the shared ones: the extension is negotiated once, on the
/// handshake, and a server may answer `tools/list` differently once it is advertised, so a headless
/// session sharing an apps-flavored connection would be told about views nobody will render.
pub fn apps_pool(
    configs: &[McpServerConfig],
    idle_reap_after: Duration,
    manifest_dir: Option<crate::tools::mcp_manifest::ManifestDir>,
) -> crate::tools::mcp_apps::McpAppsPool {
    let jobs: Vec<(McpServerConfig, Dial)> = configs
        .iter()
        .map(|config| {
            (
                config.clone(),
                Dial {
                    apps: true,
                    ..Dial::default()
                },
            )
        })
        .collect();
    // Its own manifest file: an apps-flavored server may advertise a different tool list.
    let manifest_dir = manifest_dir.map(|d| d.for_apps());
    crate::tools::mcp_apps::McpAppsPool::new(move |only: Option<Vec<String>>| {
        let jobs: Vec<_> = jobs
            .iter()
            .filter(|(c, _)| only.as_ref().is_none_or(|o| o.contains(&c.name)))
            .cloned()
            .collect();
        let manifest_dir = manifest_dir.clone();
        Box::pin(async move { connect_each(&jobs, idle_reap_after, manifest_dir.as_ref()).await })
    })
}

/// One server's dial outcome: its tools and catalog, or why it failed.
pub(crate) type ServerConnect = (
    String,
    Result<(Vec<Arc<dyn Tool>>, McpServerCatalog), String>,
);

/// Connect **one session's own** MCP connectors, named and credentialed by its
/// [session grant](crate::grant), and isolated from every other session on this replica.
///
/// Four differences from [`connect_all`], each a way a host-wide assumption would otherwise leak
/// across tenants:
///
/// - **Headers are pre-resolved.** `secrets` come straight off the grant into [`SecretHeader`] and
///   never through `resolve_config_value` — see that type's doc comment for what a `!command` header
///   would do on a replica.
/// - **No OAuth.** [`ServerAuth::load`](crate::tools::mcp_oauth::ServerAuth::load) keys the host's own credential store *by server name*, so a
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
    let (jobs, refused, mut warnings) =
        granted_jobs(connectors, secrets, egress, host, false, None);
    warnings.extend(
        refused
            .into_iter()
            .map(|(name, e)| format!("mcp connector `{name}`: {e}")),
    );
    let (tools, catalog, connect_warnings) = connect_many(&jobs, idle_reap_after, None).await;
    warnings.extend(connect_warnings);
    (tools, catalog, warnings)
}

/// [`connect_granted`], per connector — the shape a session's apps pool retries failed connectors
/// in (`only` restricts it to those). A connector the egress policy refuses is a failure too, so the
/// session's client is told rather than the connector silently missing.
pub(crate) async fn connect_granted_each(
    connectors: &[crate::grant::McpConnector],
    secrets: &BTreeMap<String, Vec<crate::grant::SecretHeader>>,
    egress: &McpEgress,
    host: Arc<McpHost>,
    idle_reap_after: Duration,
    only: Option<&[String]>,
) -> Vec<ServerConnect> {
    let (jobs, refused, warnings) = granted_jobs(connectors, secrets, egress, host, true, only);
    for warning in &warnings {
        tracing::warn!(%warning, "MCP Apps: a connector header was refused");
    }
    let mut out: Vec<ServerConnect> = refused.into_iter().map(|(n, e)| (n, Err(e))).collect();
    out.extend(connect_each(&jobs, idle_reap_after, None).await);
    out
}

/// A grant's connectors as dial jobs: `(jobs, refused by egress, header warnings)`.
#[allow(clippy::type_complexity)]
fn granted_jobs(
    connectors: &[crate::grant::McpConnector],
    secrets: &BTreeMap<String, Vec<crate::grant::SecretHeader>>,
    egress: &McpEgress,
    host: Arc<McpHost>,
    apps: bool,
    only: Option<&[String]>,
) -> (
    Vec<(McpServerConfig, Dial)>,
    Vec<(String, String)>,
    Vec<String>,
) {
    let mut jobs = Vec::with_capacity(connectors.len());
    let mut refused = Vec::new();
    let mut warnings = Vec::new();
    for connector in connectors {
        if only.is_some_and(|o| !o.contains(&connector.name)) {
            continue;
        }
        if let Err(e) = egress.check(&connector.url) {
            refused.push((connector.name.clone(), e));
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
                events: Vec::new(),
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
                apps,
                skills_changed: Arc::default(),
            },
        ));
    }
    (jobs, refused, warnings)
}

/// Dial every job concurrently and fold the results — each connect is independent I/O with no data
/// dependency on any other, so connecting one at a time would add every server's latency to startup
/// instead of paying only the slowest.
async fn connect_many(
    jobs: &[(McpServerConfig, Dial)],
    idle_reap_after: Duration,
    manifest_dir: Option<&crate::tools::mcp_manifest::ManifestDir>,
) -> (Vec<Arc<dyn Tool>>, McpCatalog, Vec<String>) {
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
    let mut catalogs = Vec::new();
    let mut warnings = Vec::new();
    for (name, result) in connect_each(jobs, idle_reap_after, manifest_dir).await {
        match result {
            Ok((server_tools, catalog)) => {
                tools.extend(server_tools);
                catalogs.push(catalog);
            }
            Err(e) => {
                tracing::warn!(
                    server = %name,
                    error = %e,
                    "failed to connect to MCP server; its tools will not be available"
                );
                warnings.push(format!("mcp server `{name}`: {e}"));
            }
        }
    }
    (tools, McpCatalog::new(catalogs), warnings)
}

/// Dial every job concurrently, keeping each server's outcome separate.
async fn connect_each(
    jobs: &[(McpServerConfig, Dial)],
    idle_reap_after: Duration,
    manifest_dir: Option<&crate::tools::mcp_manifest::ManifestDir>,
) -> Vec<ServerConnect> {
    let results = futures::future::join_all(
        jobs.iter()
            .map(|(config, dial)| connect_one(config, dial, idle_reap_after, manifest_dir)),
    )
    .await;
    jobs.iter()
        .map(|(config, _)| config.name.clone())
        .zip(results)
        .collect()
}

/// Dial one server and complete the MCP handshake, without listing anything. Split out of
/// [`connect_one`] so [`McpConnection`] can redial the exact same way after a reap — a reconnect must
/// not drift from the original connect.
/// A live client, plus the process group to sweep when it is dropped. HTTP servers have no group —
/// there is no process of ours to reap.
async fn connect_one_client(
    config: &McpServerConfig,
    dial: &Dial,
) -> Result<(McpClient, ServerProc), String> {
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
    manifest_dir: &crate::tools::mcp_manifest::ManifestDir,
) -> (Vec<Arc<dyn Tool>>, McpServerCatalog) {
    let conn = Arc::new(McpConnection::dormant(
        config.clone(),
        dial.clone(),
        idle_reap_after,
    ));
    register_for_reaping(&conn, idle_reap_after);

    let apps = dial.apps.then(|| {
        crate::tools::mcp_apps::AppServer::new(
            McpServerHandle {
                name: config.name.clone(),
                conn: conn.clone(),
            },
            manifest
                .tools
                .iter()
                .map(|t| (t.remote_name.clone(), t.ui.clone().unwrap_or_default())),
            manifest
                .resources
                .iter()
                .filter_map(|r| r.size.map(|size| (r.uri.clone(), size))),
        )
    });
    let mut tools: Vec<Arc<dyn Tool>> = manifest
        .tools
        .into_iter()
        .filter(|t| t.ui.as_ref().is_none_or(|ui| ui.model_visible()))
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
        // A `ui://` resource is an MCP App's HTML: a renderer's input, never the model's.
        if crate::tools::mcp_apps::is_ui_uri(&resource.uri) {
            continue;
        }
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

    let diagnostics = manifest.skill_diagnostics;
    let skills = manifest.skills.map(|entries| {
        crate::tools::mcp_skills::attach(
            &config.name,
            &conn,
            crate::tools::mcp_skills::Listing::cached(entries, diagnostics),
            &mut tools,
            Some(manifest_dir),
        )
    });
    let catalog = McpServerCatalog {
        name: config.name.clone(),
        conn: Arc::downgrade(&conn),
        resources: resource_infos,
        prompts: prompt_infos,
        protocol_version: None,
        apps,
        skills,
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
    if let Some(dir) = manifest_dir
        && let Some(manifest) = crate::tools::mcp_manifest::load(dir, config)
    {
        tracing::debug!(
            server = %config.name,
            tools = manifest.tools.len(),
            "advertising MCP tools from the cached manifest; not starting the server"
        );
        return Ok(tools_from_manifest(
            config,
            dial,
            manifest,
            idle_reap_after,
            dir,
        ));
    }
    let (client, proc) = connect_one_client(config, dial).await?;
    tools_from_client(config, dial, client, proc, idle_reap_after, manifest_dir).await
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
/// The group is what makes the transport's final sweep (`mcp_stdio::stdio_transport`) able to catch them, and it is why the pid is read
/// here and carried on the connection.
///
/// Dropping the client is still the primary shutdown, not the kill: the MCP stdio contract is that a
/// server watches its stdin for EOF, and the OS closes this end of that pipe however this process
/// exits — a clean return, `std::process::exit`, or a fatal signal. A spec-compliant server (this
/// crate's own `mcp_fixture_stdio_server` test fixture included) sees the EOF and exits on its own,
/// and a well-behaved browser server closes its browser on the way out. The group sweep is the
/// backstop for everything that doesn't.
///
/// `stderr` is inherited (`Stdio::inherit()`, set in `mcp_events::stdio_transport`), not captured — a
/// deliberate choice, not an oversight: a server that fails to start or crashes typically explains why
/// on its own stderr, and inheriting it means that reaches the operator's own console (this process's
/// stderr) immediately, the same way a connect failure's `tracing::warn!` does.
async fn connect_stdio(
    config: &McpServerConfig,
    dial: &Dial,
    command: &str,
    args: &[String],
) -> Result<(McpClient, ServerProc), String> {
    let env = config.resolved_env();
    let cmd = tokio::process::Command::new(command).configure(|cmd| {
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
    });
    // Spawned here rather than by rmcp's `TokioChildProcess` — for every stdio server — so its
    // stdout passes through `mcp_stdio::stdio_transport`, which keeps rmcp 3.x from silently dropping
    // custom results (see `mcp_stdio::rescue`). When the transport or the connection goes away,
    // `mcp_stdio::retire` closes the server's stdin, gives it `mcp_stdio::SHUTDOWN_GRACE` to exit,
    // and then kills what is left of its group.
    let (proc, transport) = crate::tools::mcp_stdio::stdio_transport(cmd)
        .map_err(|e| format!("failed to spawn `{command}`: {e}"))?;

    let client = McpHandler::for_dial(&config.name, dial)
        .serve_with_lifecycle(transport, client_lifecycle())
        .await
        .map_err(|e| format!("MCP handshake over stdio failed: {e}"))?;
    Ok((client, Some(proc)))
}

/// The other way a server goes away: the last tool holding the connection is dropped (a `serve`
/// registry rebuild, a finished `run`). [`mcp_stdio::retire`](crate::tools::mcp_stdio::retire)
/// closes the server's stdin at once, gives it its grace and sweeps its group, on a tracked OS
/// thread. A connection never dropped before the process exits is retired by
/// `mcp_stdio::retire_all`, which every exit path of `run` and `serve` calls before waiting for the
/// sweeps.
///
/// What this does *not* cover is the agent being hard-killed: a server in its own group no longer
/// receives the terminal's signals, and nothing then sweeps it. No worse than before — an orphaned
/// browser already outlived a killed agent — and fixing it properly needs a supervisor.
impl Drop for McpConnection {
    fn drop(&mut self) {
        // `get_mut` rather than a lock: we hold `&mut self`, so no one else can be holding it.
        if let Some(live) = self.client.get_mut().take() {
            let proc = live.proc.clone();
            drop(live);
            if let Some(proc) = &proc {
                crate::tools::mcp_stdio::retire(proc);
            }
        }
    }
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
    let mut auth = None;
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
            // token attached — and refreshed again mid-session if the server rejects it (see
            // `mcp_oauth`); a server nobody has logged into (the common case — most MCP servers
            // need no auth at all, or use `headers` above for a static credential) connects exactly
            // as before.
            auth = crate::tools::mcp_oauth::ServerAuth::load(&config.name, url).await;
            agent_core::ensure_provider();
            reqwest::Client::new()
        }
    };

    let bearer_token = match &auth {
        Some(auth) => auth.token().await,
        None => None,
    };
    // No `auth_header` here: the OAuth layer attaches the server's *current* token to each request.
    let transport_config = StreamableHttpClientTransportConfig::with_uri(url.to_string())
        .custom_headers(custom_headers);
    // Wrapped so extension results survive rmcp's result decoding — see `mcp_wire` — so an MCP
    // App view's read is refused over its cap without being read whole — see `mcp_view_http` — and
    // so a rejected OAuth token is refreshed and the request retried once — see `mcp_oauth`.
    let transport = StreamableHttpClientTransport::with_client(
        crate::tools::mcp_oauth::OAuthHttp::new(
            crate::tools::mcp_view_http::ViewCappedHttp::new(crate::tools::mcp_wire::HttpClient(
                client,
            )),
            auth,
        ),
        transport_config,
    );

    McpHandler::for_dial(&config.name, dial)
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
        let _call = client
            .service()
            .track_call(calling_host(&client.service().host));
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
        let _call = client
            .service()
            .track_call(calling_host(&client.service().host));
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
    /// Set only on a connection that advertised MCP Apps: every tool's `_meta.ui`, and a strong
    /// handle for the app bridge. See [`crate::tools::mcp_apps`].
    pub(crate) apps: Option<Arc<crate::tools::mcp_apps::AppServer>>,
    /// The server's SEP-2640 skills, when it declares the extension — see
    /// [`crate::tools::mcp_skills`].
    pub(crate) skills: Option<Arc<crate::tools::mcp_skills::ServerSkills>>,
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
    /// The session this view of the catalog belongs to (see [`Self::for_session`]): the host its
    /// non-tool requests (`events/*`, `completion/complete`) are registered under, so a nested
    /// request raised during one is attributed to that session or refused, never guessed.
    session_host: Option<Arc<McpHost>>,
}

impl McpCatalog {
    pub fn new(servers: Vec<McpServerCatalog>) -> Self {
        Self {
            servers: Arc::new(std::sync::Mutex::new(servers)),
            session_host: None,
        }
    }

    /// This catalog (the same servers, shared) as session `host` uses it.
    pub fn for_session(&self, host: Arc<McpHost>) -> Self {
        Self {
            servers: self.servers.clone(),
            session_host: Some(host),
        }
    }

    /// The session host this view was made [for](Self::for_session), if any.
    pub(crate) fn session_host(&self) -> Option<Arc<McpHost>> {
        self.session_host.clone()
    }

    /// The host a request made through this catalog belongs to.
    fn request_host(&self, client: &McpClient) -> Arc<McpHost> {
        self.session_host
            .clone()
            .unwrap_or_else(|| calling_host(&client.service().host))
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

    /// Every configured server's declared MCP Events subscriptions (`mcp_servers[].events`), for
    /// the servers still connected. Read off the connection's own config, so it is exactly what was
    /// dialed — nothing is re-read from disk.
    pub fn event_subscriptions(&self) -> Vec<(String, Vec<crate::settings::McpEventSubscription>)> {
        self.snapshot()
            .into_iter()
            .filter_map(|s| {
                let conn = s.conn.upgrade()?;
                (!conn.config.events.is_empty()).then(|| (s.name, conn.config.events.clone()))
            })
            .collect()
    }

    /// Whether any server is connected at all — the cheap gate `serve` uses before wiring events.
    pub fn is_empty(&self) -> bool {
        self.snapshot().is_empty()
    }

    /// How to reach `server` for MCP Events requests, redialing a reaped process first.
    ///
    /// - **stdio**, and a pre-`2026-07-28` (session-bound) HTTP server: rmcp's own connection. The handle keeps the process alive (the reaper never
    ///   reaps a client somebody holds), so an open push stream pins its server while it is open.
    /// - **streamable HTTP**: the URL, credential headers and negotiated protocol version, for
    ///   direct stateless requests — rmcp's typed result union would drop a custom result that
    ///   carries `_meta` (see `mcp_events::rescue_line`), and over HTTP there is no byte stream of
    ///   ours to repair it on.
    pub(crate) async fn events_peer(&self, server: &str) -> Result<EventsPeer, String> {
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
        // Direct requests only for a stateless (`2026-07-28`) server. An older streamable-HTTP
        // server may bind requests to the `Mcp-Session-Id` rmcp negotiated — only rmcp's own
        // connection carries it — and it does not attach `_meta` to results, so rmcp's own path
        // decodes them intact.
        let stateless = client
            .peer_info()
            .is_some_and(|i| i.protocol_version >= ProtocolVersion::V_2026_07_28);
        let (McpTransport::Http { url, .. }, true) = (&conn.config.transport, stateless) else {
            return Ok(EventsPeer::Rmcp {
                peer: client.peer().clone(),
                router: client.service().events.clone(),
                host: self.request_host(&client),
                live: ClientHold(client),
            });
        };
        let mut headers: Vec<(HeaderName, HeaderValue)> = Vec::new();
        let mut auth = None;
        match &conn.dial.http {
            Some(http) => headers.extend(
                http.headers
                    .iter()
                    .map(|h| (h.name.clone(), h.value.clone())),
            ),
            None => {
                for (k, v) in conn.config.resolved_headers() {
                    if let (Ok(k), Ok(v)) = (
                        HeaderName::from_bytes(k.as_bytes()),
                        HeaderValue::from_str(&v),
                    ) {
                        headers.push((k, v));
                    }
                }
                auth = crate::tools::mcp_oauth::ServerAuth::load(&conn.config.name, url).await;
                if let Some(auth) = &auth
                    && let Some(token) = auth.token().await
                    && let Ok(v) = HeaderValue::from_str(&format!("Bearer {token}"))
                {
                    headers.push((http::header::AUTHORIZATION, v));
                }
            }
        }
        Ok(EventsPeer::Http {
            url: url.clone(),
            router: client.service().events.clone(),
            headers,
            auth,
            protocol_version: client
                .peer_info()
                .map(|i| i.protocol_version.to_string())
                .unwrap_or_else(|| ProtocolVersion::V_2026_07_28.to_string()),
        })
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
        let _call = client.service().track_call(self.request_host(&client));
        client
            .complete(params)
            .await
            .map_err(|e| format!("mcp server `{server}` completion/complete failed: {e}"))
    }
}

/// Keeps a client (and so its process) alive; deliberately opaque outside this module.
pub(crate) struct ClientHold(Arc<McpClient>);

/// One server's connection, as the MCP Events client needs it. See [`McpCatalog::events_peer`].
pub(crate) enum EventsPeer {
    /// stdio: the raw peer for `events/*` requests, the router its `notifications/events/*` arrive
    /// on, and a hold on the client so the idle reaper leaves it alone while this is alive.
    Rmcp {
        peer: Peer<RoleClient>,
        router: crate::tools::mcp_events::NotificationRouter,
        /// The session the requests belong to (see [`EventsPeer::track_request`]).
        host: Arc<McpHost>,
        live: ClientHold,
    },
    /// Streamable HTTP: where and how to POST `events/*` directly. `router` still carries
    /// `notifications/events/list_changed`, which arrives on rmcp's own connection.
    Http {
        url: String,
        router: crate::tools::mcp_events::NotificationRouter,
        headers: Vec<(HeaderName, HeaderValue)>,
        /// The server's OAuth login, if any: a 401 refreshes it and the request is retried once
        /// (see `mcp_oauth`).
        auth: Option<Arc<crate::tools::mcp_oauth::ServerAuth>>,
        protocol_version: String,
    },
}

impl EventsPeer {
    /// Register one `events/*` request as in flight for its session, for as long as the returned
    /// guard lives, so a nested request the server raises during it is attributed or refused like
    /// one raised during a `tools/call` (see `McpHandler::route`). `None` over direct HTTP, where
    /// rmcp never sees the exchange (and so never delivers a nested request from it).
    pub(crate) fn track_request(&self) -> Option<EventsCall> {
        match self {
            Self::Rmcp { host, live, .. } => {
                Some(EventsCall(live.0.service().track_call(host.clone())))
            }
            Self::Http { .. } => None,
        }
    }
}

/// An `events/*` request registered as in flight (see [`EventsPeer::track_request`]).
pub(crate) struct EventsCall(ActiveCallGuard);

impl EventsCall {
    /// Record the request's id once sent, so a nested request on its stream routes to it.
    pub(crate) fn bind(&self, request: RequestId) {
        self.0.bind(request);
    }
}

async fn tools_from_client(
    config: &McpServerConfig,
    dial: &Dial,
    client: McpClient,
    // The server's process, so a reap can take its group too; `None` for HTTP.
    proc: ServerProc,
    idle_reap_after: Duration,
    manifest_dir: Option<&crate::tools::mcp_manifest::ManifestDir>,
) -> Result<(Vec<Arc<dyn Tool>>, McpServerCatalog), String> {
    let remote_tools = client
        .list_all_tools()
        .await
        .map_err(|e| format!("`tools/list` failed: {e}"))?;
    // Fail-soft: a server without resources/prompts capabilities returns an error; treat as empty.
    let mut remote_resources = match client.list_all_resources().await {
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
    // SEP-2640 skills: the listing only — no skill file is read at connect (see `mcp_skills`). A
    // declared extension's files are reached through the verified skill tool, so they are not also
    // wrapped as generic resource tools: a tool per file is context spent every turn on an
    // unverified second route to the same bytes.
    let skill_listing = match crate::tools::mcp_skills::declared(&client) {
        true => Some(crate::tools::mcp_skills::discover(&client, &config.name).await),
        false => None,
    };
    if let Some(listing) = &skill_listing {
        let roots = crate::tools::mcp_skills::skill_roots_among(
            remote_resources.iter().map(|r| r.uri.as_str()),
        );
        remote_resources.retain(|r| {
            !listing.hides(&r.uri)
                && !roots.iter().any(|root| {
                    r.uri
                        .strip_prefix(root.as_str())
                        .is_some_and(|rest| rest.starts_with('/'))
                })
        });
    }
    let protocol_version = client
        .peer_info()
        .map(|info| info.protocol_version.to_string());

    let conn = Arc::new(McpConnection::new(
        config.clone(),
        dial.clone(),
        client,
        proc,
        idle_reap_after,
    ));
    register_for_reaping(&conn, idle_reap_after);

    // Record what this server advertises so the *next* boot can skip starting it entirely. Written
    // after a successful `tools/list`, so a server that failed to enumerate never poisons the cache.
    //
    // A skills listing marked `cacheScope: "private"` must not be served outside the authorization
    // context it was fetched in, and this file outlives that context (a rotated token, a changed
    // header). Such a server is forgotten instead, so the next boot asks it again.
    let private_listing = skill_listing.as_ref().is_some_and(|l| l.private);
    if let (Some(dir), true) = (manifest_dir, private_listing) {
        crate::tools::mcp_manifest::forget(dir, config).await;
    }
    if let (Some(dir), false) = (manifest_dir, private_listing) {
        crate::tools::mcp_manifest::store(
            dir,
            config,
            remote_tools
                .iter()
                .map(|t| crate::tools::mcp_manifest::CachedTool {
                    remote_name: t.name.to_string(),
                    description: t.description.as_deref().unwrap_or_default().to_string(),
                    input_schema: t.schema_as_json_value(),
                    // Both flavors: a server may send `_meta.ui` whether or not it was negotiated
                    // (the official SDK's `registerAppTool` always does), and an app-only tool
                    // must stay hidden from the model either way.
                    ui: crate::tools::mcp_apps::ToolUi::from_meta(t.meta.as_deref()),
                })
                .collect(),
            remote_resources
                .iter()
                .map(|r| crate::tools::mcp_manifest::CachedResource {
                    name: r.name.clone(),
                    uri: r.uri.clone(),
                    size: r.size,
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
            skill_listing.as_ref().map(|l| l.entries.clone()),
            skill_listing
                .as_ref()
                .map(|l| l.diagnostics.clone())
                .unwrap_or_default(),
        )
        .await;
    }

    let apps = dial.apps.then(|| {
        crate::tools::mcp_apps::AppServer::new(
            McpServerHandle {
                name: config.name.clone(),
                conn: conn.clone(),
            },
            remote_tools.iter().map(|t| {
                (
                    t.name.to_string(),
                    crate::tools::mcp_apps::ToolUi::from_meta(t.meta.as_deref())
                        .unwrap_or_default(),
                )
            }),
            remote_resources
                .iter()
                .filter_map(|r| r.size.map(|size| (r.uri.clone(), size))),
        )
    });
    let mut tools: Vec<Arc<dyn Tool>> = remote_tools
        .into_iter()
        // Spec: "Host MUST NOT include tools in the agent's tool list when their visibility does not
        // include `model`" — whichever flavor of connection listed them.
        .filter(|t| {
            crate::tools::mcp_apps::ToolUi::from_meta(t.meta.as_deref())
                .is_none_or(|ui| ui.model_visible())
        })
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
        // A `ui://` resource is an MCP App's HTML: a renderer's input, never the model's.
        if crate::tools::mcp_apps::is_ui_uri(&resource.uri) {
            continue;
        }
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

    let skills = skill_listing.map(|listing| {
        crate::tools::mcp_skills::attach(&config.name, &conn, listing, &mut tools, manifest_dir)
    });
    Ok((
        tools,
        McpServerCatalog {
            name: config.name.clone(),
            conn: Arc::downgrade(&conn),
            resources: resource_infos,
            prompts: prompt_infos,
            protocol_version,
            apps,
            skills,
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
        assert_eq!(
            answered_by(&a.elicit(params.clone(), None).await.unwrap()),
            "session-a"
        );
        assert_eq!(
            answered_by(&b.elicit(params.clone(), None).await.unwrap()),
            "session-b"
        );
        // And a connection built the process-wide way still reaches the process-wide hub.
        let shared = McpHandler::new("linear", host());
        assert_eq!(
            answered_by(&shared.elicit(params, None).await.unwrap()),
            "process"
        );
    }

    /// N1: on a connection shared by sessions, a nested request goes to the session whose call is
    /// in flight when that is unambiguous (one session, or the HTTP stream names the call), and is
    /// refused, reaching nobody, when calls from two sessions are in flight and nothing says which.
    #[tokio::test]
    async fn a_nested_request_is_routed_only_when_attributable() {
        let session_a = Arc::new(McpHost::new());
        let session_b = Arc::new(McpHost::new());
        session_a
            .elicitation
            .install(Arc::new(MarkerGate("session-a")));
        session_b
            .elicitation
            .install(Arc::new(MarkerGate("session-b")));
        let shared = McpHandler::new("linear", host());
        let params = ElicitRequestParams::UrlElicitationParams {
            meta: None,
            message: "which session?".into(),
            url: "https://example.com/ask".into(),
            elicitation_id: "e1".into(),
        };
        let call_a = shared.track_call(session_a.clone());
        call_a.bind(RequestId::Number(7));
        let second_a = shared.track_call(session_a.clone());
        assert_eq!(
            answered_by(&shared.elicit(params.clone(), None).await.unwrap()),
            "session-a",
            "every call in flight is A's"
        );
        let call_b = shared.track_call(session_b.clone());
        call_b.bind(RequestId::Number(8));
        let refused = shared.elicit(params.clone(), None).await.unwrap_err();
        assert!(
            refused.message.contains("more than one session"),
            "{refused:?}"
        );
        let origin = InboundStreamOrigin::OutboundRequest(RequestId::Number(7));
        assert_eq!(
            answered_by(&shared.elicit(params.clone(), Some(&origin)).await.unwrap()),
            "session-a",
            "the HTTP stream names A's call"
        );
        drop((call_a, second_a));
        assert_eq!(
            answered_by(&shared.elicit(params, None).await.unwrap()),
            "session-b",
            "only B's call is left"
        );
        drop(call_b);
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
