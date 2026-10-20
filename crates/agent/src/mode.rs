//! Which world a `serve` session lives in: this host, or one tenant's slice of a multi-tenant
//! replica.
//!
//! `serve --service` makes the daemon a fail-closed multi-tenant service (see [`crate::service`]).
//! Everything `serve_session` does differently there is a *rule* about where a thing comes from or
//! where it may go: the tenant's directory instead of the replica's, the sandbox instead of this
//! disk, the grant instead of the operator's credentials. There are exactly two worlds, so this is
//! an enum with one method per rule rather than a trait: a reader asking "what does service mode
//! change about X?" greps for the method, and the `Local` arm of each is the behaviour `serve` had
//! before service mode existed.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::exec_endpoint::ExecCell;
use crate::serve::ServeConfig;
use crate::service::{Resources, ServiceSession, Snapshot};
use crate::session_store::{Layout, RepoOptions, SessionRepo, TenantCodec};

/// Decided once, at startup, from the flags; a daemon then hands each session its own `Service`.
#[derive(Clone)]
pub enum Mode {
    /// This host: its disk, its credentials, its `$HOME`.
    Local,
    /// A `serve --service` process before any connection's grant has bound a session to a tenant:
    /// what [`ServeConfig`] carries at startup, and what a session that never got a grant would
    /// see. Every rule is the service one, and [`Self::for_persistence`] refuses to open it, so
    /// nothing tenant-shaped can land on the replica's own disk.
    Replica,
    /// One tenant on a shared replica, fixed by the connection's verified grant.
    Service(Arc<ServiceSession>),
}

impl Mode {
    /// The process's mode, from whether `--service` was given.
    pub fn process(service: bool) -> Self {
        if service { Mode::Replica } else { Mode::Local }
    }

    /// This mode bound to `service`'s grant when there is one. A replica's session without a grant
    /// stays a `Replica`, which fails closed.
    pub(crate) fn bound_to(&self, service: Option<Arc<ServiceSession>>) -> Self {
        match service {
            Some(service) => Mode::Service(service),
            None => self.clone(),
        }
    }

    fn is_local(&self) -> bool {
        matches!(self, Mode::Local)
    }

    /// The mode persistence opens in. Fail-closed: a `Replica` that reached a session without a
    /// grant would otherwise land in the per-cwd default repo, the replica's own disk, shared
    /// across tenants.
    pub(crate) fn for_persistence(cfg: &ServeConfig) -> std::io::Result<Self> {
        match &cfg.mode {
            Mode::Replica => Err(std::io::Error::other(
                "service mode: this session has no verified grant, so it has nowhere to persist",
            )),
            mode => Ok(mode.clone()),
        }
    }

    /// The session's own service handle, for the few places that drive the grant's machinery
    /// directly (attaching its sandbox, registering its job notifications).
    fn service(&self) -> Option<&Arc<ServiceSession>> {
        match self {
            Mode::Local | Mode::Replica => None,
            Mode::Service(service) => Some(service),
        }
    }

    // ---- the process ----------------------------------------------------------------------

    /// Whether an unconfigured async worker count is one per core (the CPU a replica's task was
    /// given) rather than a single thread.
    pub fn scales_workers_with_cores(&self) -> bool {
        !self.is_local()
    }

    /// Whether `--notify-origin` means anything: job notify is a service-mode feature.
    pub fn offers_job_notify(&self) -> bool {
        !self.is_local()
    }

    /// Whether startup demands a grant keyring, shards and a listener, and refuses the flags that
    /// name host resources a tenant must not reach.
    pub fn refuses_host_resources(&self) -> bool {
        !self.is_local()
    }

    /// Whether the operator's own `mcp_servers` are connected, once, for every session.
    pub fn connects_operator_mcp(&self) -> bool {
        self.is_local()
    }

    /// The one client every session's own connectors dial through, in service mode. Its egress
    /// policy is fixed at build time, which is why it is a process flag rather than anything a
    /// grant can ask for.
    pub fn session_mcp_egress(
        &self,
        allow_private: bool,
    ) -> Result<Option<crate::tools::mcp::McpEgress>, String> {
        if self.is_local() {
            return Ok(None);
        }
        crate::tools::mcp::McpEgress::new(crate::tools::web::ssrf::EgressPolicy::new(
            allow_private,
            &[],
        ))
        .map(Some)
    }

    /// The ambient provider env a session may read. None in service mode: `AI_DIRECT`/
    /// `ANTHROPIC_API_KEY` and friends are this replica's environment, and a session's only
    /// credential is the one sealed in its grant.
    pub fn provider_env(&self, gateway_configured: bool) -> crate::gateway_credential::ProviderEnv {
        if self.is_local() {
            crate::gateway_credential::ProviderEnv::from_process_env(gateway_configured)
        } else {
            crate::gateway_credential::ProviderEnv::default()
        }
    }

    // ---- the daemon -----------------------------------------------------------------------

    /// Whether every connection must present a verified grant. Otherwise the daemon is the
    /// operator's own: loopback, unauthenticated, with the operator's MCP Events and session
    /// directory.
    pub(crate) fn verifies_grants(&self) -> bool {
        !self.is_local()
    }

    /// Whether the daemon owns MCP Events: the webhook route, the receiver document and the
    /// events session. Never in service mode.
    pub(crate) fn serves_mcp_events(&self) -> bool {
        self.is_local()
    }

    /// Whether the daemon lists from one process-wide session directory. A replica lists per
    /// tenant, from the shards.
    pub(crate) fn has_process_session_dir(&self) -> bool {
        self.is_local()
    }

    /// The grant's Code Mode claim. Code Mode is a per-process flag everywhere else, which service
    /// mode refuses: on a replica one tenant's setting would be every tenant's.
    pub(crate) fn code_mode_claim(&self) -> Option<bool> {
        self.service().map(|service| service.code_mode())
    }

    // ---- the sandbox's exec backend ---------------------------------------------------------

    /// How the grant's sandbox is dialed, on first use. `None` outside service mode, where the
    /// exec target comes from flags or the session's persisted endpoint.
    pub(crate) fn lazy_exec(
        &self,
        max_response_bytes: usize,
    ) -> Option<crate::exec_endpoint::Connect> {
        self.service()
            .map(|service| service.lazy_exec(max_response_bytes))
    }

    /// The sandbox's `$HOME` as last seen, once probed.
    pub(crate) fn sandbox_home(&self) -> Option<String> {
        self.service()
            .and_then(|service| service.sandbox())
            .map(|sandbox| sandbox.home)
    }

    /// Hand the grant what a saved snapshot learned of its sandbox, so it need not be probed again.
    pub(crate) fn seed_sandbox(&self, sandbox: crate::service::Sandbox) {
        if let Some(service) = self.service() {
            service.seed_sandbox(sandbox);
        }
    }

    /// Register this session for the sandbox's `job_ended` callbacks, if the replica runs a hub
    /// and this is a service session.
    pub(crate) fn register_job_notify(
        &self,
        hub: Option<&Arc<crate::tools::job_notify::NotifyHub>>,
        session_id: &str,
    ) -> Option<Arc<crate::tools::job_notify::SessionNotify>> {
        let (hub, service) = (hub?, self.service()?);
        Some(crate::tools::job_notify::SessionNotify::register(
            hub,
            service.tenant(),
            session_id,
        ))
    }

    /// Whose session this is. One replica emits for many tenants; `None` outside service mode.
    pub(crate) fn tenant(&self) -> Option<&str> {
        self.service().map(|service| service.tenant())
    }

    // ---- commands -------------------------------------------------------------------------

    /// Why `command` is refused outright, if it is. See [`crate::service::refused_command`].
    pub(crate) fn refused_command(&self, command: &str) -> Option<&'static str> {
        match self {
            Mode::Local => None,
            Mode::Replica | Mode::Service(_) => crate::service::refused_command(command),
        }
    }

    /// A `fork`/`clone` arriving while a prompt is `activity` (running, retrying): in service mode
    /// these two derive a *new* session and leave this one running (see
    /// [`Self::forks_detached`]), so there is nothing here worth a run for. "Busy" costs the client
    /// a retry; cancelling would cost it the turn.
    pub(crate) fn busy_refusal_for_fork_clone(
        &self,
        command: &str,
        activity: &str,
    ) -> Option<String> {
        match (self, command) {
            (Mode::Replica | Mode::Service(_), "fork" | "clone") => Some(format!(
                "busy: a prompt is {activity}; fork/clone copy what is on disk, so retry once the run has finished"
            )),
            _ => None,
        }
    }

    /// `fork`/`clone` derive a new session without switching to it: this task keeps its own id, its
    /// lock and its grant, and the client gets the new id to connect to.
    pub(crate) fn forks_detached(&self) -> bool {
        !self.is_local()
    }

    /// `export_html` returns the document inline: writing a file would put a whole transcript on
    /// the replica's disk, and there is no per-tenant filesystem to write it to instead.
    pub(crate) fn exports_inline(&self) -> bool {
        !self.is_local()
    }

    // ---- what the replica's own disk may decide -----------------------------------------------

    /// Whether the replica's filesystem may set the base prompt (`SYSTEM.md`/`APPEND_SYSTEM.md`).
    /// A tenant's own arrive from its sandbox instead.
    pub(crate) fn reads_disk_overrides(&self) -> bool {
        self.is_local()
    }

    /// Whether the project-trust gate applies. A sandbox's contents are the one tenant's own, so
    /// there is no second party for the gate to protect against.
    pub(crate) fn trust_gate_applies(&self) -> bool {
        self.is_local()
    }

    /// Whether a `--model` flag is only this process's default rather than the operator's choice for
    /// the session. On a replica it is the former: one flag serves every tenant, so a tenant's own
    /// `AGENTS.md` model outranks it.
    pub(crate) fn model_flag_is_a_default(&self) -> bool {
        !self.is_local()
    }

    /// The grant's ceilings narrowing `tools` and `max_steps`. Outside service mode there is no grant,
    /// so both come back as given.
    pub(crate) fn apply_limits(
        &self,
        tools: Option<Vec<String>>,
        max_steps: Option<u32>,
    ) -> (Option<Vec<String>>, Option<u32>) {
        match self.service() {
            Some(service) => (
                service.limits().narrow_tools(tools),
                service.limits().cap_steps(max_steps),
            ),
            None => (tools, max_steps),
        }
    }

    /// Why `model` may not run in this session, if the grant's `limits` forbid it. The model is
    /// refused, never swapped for another.
    pub(crate) fn model_refusal(&self, model: &str) -> Option<String> {
        self.service()?.limits().check_model(model).err()
    }

    /// Whether to reap subagent worktrees orphaned by a crash. Not in service mode: the only repo
    /// that could sweep is the replica's own checkout.
    pub(crate) fn sweeps_worktrees(&self) -> bool {
        self.is_local()
    }

    /// Whether the operator's persisted settings are read and written. In service mode the store
    /// lives under the replica's `$HOME`: reading it would let the operator's machine configure a
    /// tenant, and writing it would let one tenant's toggle reach every other session.
    pub fn reads_operator_settings(&self) -> bool {
        self.is_local()
    }

    /// Whether MCP Events apply. Never in service mode: a grant connector has no `events`
    /// configuration, and the `mcp_events_*` commands are refused there.
    pub(crate) fn allows_mcp_events(&self) -> bool {
        self.is_local()
    }

    /// `models.json` header overrides for `model`. Empty in service mode: that file is the
    /// replica operator's own.
    pub(crate) fn extra_headers(&self, model: &str) -> std::collections::HashMap<String, String> {
        match self {
            Mode::Local => crate::serve::model_override_extra_headers(model),
            Mode::Replica | Mode::Service(_) => std::collections::HashMap::new(),
        }
    }

    /// The only credential service mode takes: the grant's sealed gateway key.
    pub(crate) fn gateway_key(&self) -> Option<String> {
        self.service()
            .map(|service| service.gateway_key().to_owned())
    }

    // ---- where the session lives --------------------------------------------------------------

    /// The directory the session's tools and prompt are rooted in: the grant's `workspace_root`
    /// (a path inside the sandbox this process never stats), or this process's cwd.
    pub(crate) fn session_cwd(&self) -> PathBuf {
        match self {
            Mode::Service(service) => PathBuf::from(service.workspace_root()),
            Mode::Local | Mode::Replica => {
                crate::session_store::canonical_cwd(&std::env::current_dir().unwrap_or_default())
            }
        }
    }

    /// What [`tools::ToolConfig::root`](crate::tools::ToolConfig::root) reads: the grant's
    /// `workspace_root`, or empty for "the process cwd".
    pub(crate) fn tool_root(&self) -> PathBuf {
        match self {
            Mode::Service(service) => PathBuf::from(service.workspace_root()),
            Mode::Local | Mode::Replica => PathBuf::new(),
        }
    }

    /// What the startup probe found on the far side (`bash` or `sh`).
    pub(crate) fn remote_shell(&self) -> Option<&'static str> {
        self.service()
            .and_then(|service| service.sandbox())
            .map(|sandbox| sandbox.shell)
    }

    /// Whether `meta_cwd` no longer reflects reality. Never stale in service mode: the recorded
    /// cwd *is* `workspace_root`, and the staleness check stats the path — on the replica, where
    /// it doesn't exist.
    pub(crate) fn cwd_is_stale(&self, meta_cwd: &str, actual_cwd: &Path) -> bool {
        self.is_local() && crate::serve::cwd_is_stale(meta_cwd, actual_cwd)
    }

    /// The current branch of the session's working directory. In service mode the lookup runs
    /// **in the sandbox**, through the session's own runner: `cwd` is a sandbox path, so asking
    /// this host about it would either fail or answer about the replica's own checkout.
    pub(crate) async fn git_branch(&self, exec: &ExecCell, cwd: &Path) -> Option<String> {
        match self {
            Mode::Service(service) => {
                crate::serve::git_branch_in(exec, service.workspace_root()).await
            }
            Mode::Local | Mode::Replica => crate::serve::git_branch(cwd).await,
        }
    }

    /// Whether `session_file` is reported. It is not in service mode: the path is a mount on the
    /// replica, which is this tenant's storage but not its filesystem.
    pub(crate) fn reports_session_file(&self) -> bool {
        self.is_local()
    }

    /// Strict in service mode: an empty cell is an error, never this host.
    pub(crate) fn new_exec_cell(&self) -> ExecCell {
        match self {
            Mode::Replica | Mode::Service(_) => ExecCell::strict(),
            Mode::Local => ExecCell::new(),
        }
    }

    // ---- redaction ----------------------------------------------------------------------------

    /// Whether failed responses have replica-host paths stripped from their `error` text.
    pub(crate) fn redacts_host_paths(&self) -> bool {
        !self.is_local()
    }

    /// `text` with replica-host paths replaced; unchanged outside service mode.
    pub(crate) fn redact(&self, text: &str) -> String {
        match self {
            Mode::Local | Mode::Replica => text.to_owned(),
            Mode::Service(service) => service.redact(text),
        }
    }

    // ---- persistence --------------------------------------------------------------------------

    /// The tenant's own session directory, its workspace root and its repo shape: the layout a
    /// mount shared by many replicas needs. `None` outside service mode.
    pub(crate) fn tenant_storage(&self) -> Option<(&Path, &str, RepoOptions)> {
        self.service().map(|service| {
            (
                service.session_dir.as_path(),
                service.workspace_root(),
                service.repo_options_for(service.session_id()),
            )
        })
    }

    /// The repo a command naming session `id` acts on. A session id carries its shard, so one
    /// tenant's sessions can live on several mounts, and a command must reach the one it named. A
    /// shard this replica doesn't serve is an error, not an empty result. `None` outside service
    /// mode, where it is always this session's own.
    pub(crate) fn tenant_repo_for(&self, id: &str) -> Option<std::io::Result<SessionRepo>> {
        let service = self.service()?;
        Some(
            service
                .session_dir_for(id)
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("session {id} is on a shard this replica does not serve"),
                    )
                })
                .and_then(|dir| SessionRepo::open_with(dir, service.repo_options_for(id))),
        )
    }

    /// Whether an id must name a session **exactly**. A tenant's ids are minted, never typed, so a
    /// unique-prefix match turns "that session is gone" into "here is a neighbour of it".
    pub(crate) fn requires_exact_ids(&self) -> bool {
        !self.is_local()
    }

    /// The directories and layout a listing of *this* session's sessions scans, replacing the cwd
    /// filter: the directory is already `<shard>/<tenant>/sessions`, scoped by construction.
    pub(crate) fn listing_scope(&self) -> Option<(Vec<PathBuf>, Layout)> {
        self.service()
            .map(|service| (vec![service.session_dir.clone()], service.layout()))
    }

    /// The directories and layout of everything this tenant owns, across every mounted shard.
    /// Also the scope of the trash. Never another tenant's, and never the sibling-project walk.
    pub(crate) fn tenant_scope(&self) -> Option<(Vec<PathBuf>, Layout)> {
        self.service()
            .map(|service| (service.tenant_session_dirs(), service.layout()))
    }

    /// The tenant's sealing key and the session id it seals under, for the app-state sidecars.
    pub(crate) fn sidecar_seal(&self, session_id: &str) -> Option<(Arc<TenantCodec>, String)> {
        self.service()
            .map(|service| (Arc::clone(service.codec()), session_id.to_string()))
    }

    // ---- memory -------------------------------------------------------------------------------

    /// The tenant's own `<home shard>/<tenant>/memory`, never a DSN and never a path derived from
    /// the replica's cwd.
    pub(crate) fn tenant_memory_backend(&self) -> Option<Arc<dyn crate::memory::MemoryBackend>> {
        self.service().map(|service| service.memory_backend())
    }

    /// Seal the per-session memory mount with the tenant's key, exactly like the durable mount and
    /// the transcript beside it.
    pub(crate) fn seal_session_memory(
        &self,
        backend: crate::memory::file::FileBackend,
    ) -> crate::memory::file::FileBackend {
        match self {
            Mode::Service(service) => backend.sealed(Arc::clone(service.codec())),
            Mode::Local | Mode::Replica => backend,
        }
    }

    // ---- the sandbox --------------------------------------------------------------------------

    /// This session's own MCP connectors, named and credentialed by its grant and dialed now, not
    /// at process startup. Fail-soft per connector. Returns whether it ran.
    pub(crate) async fn connect_session_mcp(
        &self,
        cfg: &mut ServeConfig,
        host: &Arc<crate::tools::mcp_host::McpHost>,
        session_id: &str,
        from_file: &[crate::grant::McpConnector],
    ) -> bool {
        let Some(service) = self.service() else {
            return false;
        };
        let (tools, catalog, warnings) = service
            .mcp(cfg.mcp_http.as_ref(), host.clone(), from_file)
            .await;
        cfg.mcp_apps = Some(service.mcp_apps_pool(cfg.mcp_http.clone(), host.clone(), from_file));
        for warning in &warnings {
            eprintln!("warning: session {session_id}: {warning}");
        }
        cfg.mcp_tools = tools;
        cfg.mcp_catalog = catalog;
        true
    }

    /// The connectors the workspace's `.mcp.json` adds, for a session that has just probed its
    /// sandbox (a reopened one takes them from its snapshot instead). Empty outside service mode.
    pub(crate) async fn read_workspace_mcp(
        &self,
        exec: &ExecCell,
        session_id: &str,
    ) -> Vec<crate::grant::McpConnector> {
        let Some(service) = self.service() else {
            return Vec::new();
        };
        let (connectors, warnings) = crate::service::workspace_mcp(
            exec.backend().as_ref(),
            std::path::Path::new(service.workspace_root()),
        )
        .await;
        for warning in &warnings {
            eprintln!("warning: session {session_id}: {warning}");
        }
        connectors
    }

    /// Whether a sandbox snapshot sits beside the transcript. When the saved one is behind what the
    /// session knows it is rewritten at the top of the loop, once the session's directory exists.
    pub(crate) fn keeps_sandbox_snapshot(&self) -> bool {
        !self.is_local()
    }

    /// What the session last learned about its sandbox, if it reopens one.
    pub(crate) fn load_sandbox_snapshot(
        &self,
        session_file: Option<&Path>,
        session_id: &str,
        context_files: bool,
    ) -> Option<Snapshot> {
        match (self, session_file) {
            (Mode::Service(service), Some(file)) => {
                service.load_snapshot(file, session_id, context_files)
            }
            _ => None,
        }
    }

    /// Discover the tenant's resources through the session's exec backend. `None` outside service
    /// mode, where discovery walks this host.
    pub(crate) async fn discover_sandbox_resources(
        &self,
        exec: &ExecCell,
        context_files: bool,
    ) -> Option<Resources> {
        let service = self.service()?;
        Some(
            service
                .resources(exec.backend().as_ref(), context_files)
                .await,
        )
    }

    /// Save the sandbox snapshot. `None` when there is nothing to save it with (not service mode,
    /// or the sandbox has not been probed yet).
    pub(crate) fn save_sandbox_snapshot(
        &self,
        session_file: &Path,
        session_id: &str,
        context_files: bool,
        resources: &Resources,
    ) -> Option<std::io::Result<()>> {
        let service = self.service()?;
        let sandbox = service.sandbox()?;
        Some(service.save_snapshot(session_file, session_id, context_files, &sandbox, resources))
    }

    /// The parent's project-instruction files when they were fetched through a backend (service
    /// mode), so a child reuses them instead of walking a `cwd` that isn't on this machine.
    pub(crate) fn context_files<'a>(
        &self,
        files: &'a [(String, String)],
    ) -> Option<&'a [(String, String)]> {
        (!self.is_local()).then_some(files)
    }
}
