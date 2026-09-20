//! Service mode: one `serve` replica, many tenants, nothing of the replica's own reaching any of
//! them.
//!
//! `serve --listen … --service` turns the daemon from "a headless agent for whoever is on the other
//! end of this socket" into a **fail-closed multi-tenant service**. Every connection presents a
//! [`bsg_v1` session grant](crate::grant) in the `x-beyond-grant` header; the grant says which tenant
//! and session it is, where that session's sandbox and workspace are, and — sealed — the credentials
//! to reach them. This module is the seam between that grant and the rest of `serve`:
//!
//! - [`authorize`] turns a token plus a `?session_id=` into a [`ServiceSession`], or into exactly one
//!   [`Refusal`] (and so exactly one HTTP status). Nothing downstream re-derives tenancy.
//! - [`ServiceSession`] then answers every "where does this session's *X* live?" question `serve`
//!   used to answer from the replica host: persistence ([`ServiceSession::session_dir`]), durable
//!   memory ([`ServiceSession::memory_backend`]), the exec target every tool runs in
//!   ([`ServiceSession::connect_exec`]), the gateway credential ([`ServiceSession::gateway_key`]),
//!   and the MCP connectors this session may reach ([`ServiceSession::mcp`]).
//!
//! **The no-host rule.** In service mode the replica's own `$HOME`, cwd, and stored settings are not
//! a fallback for anything: no `~/.claude` skills/prompts/agents/`SYSTEM.md`, no `models.json` or
//! `auth.json`, no per-cwd session directory, no `RealRunner`/`LocalFs` behind an unset exec
//! endpoint. Where a host default would have applied, service mode **refuses** instead — see
//! [`refused_command`] and [`ExecCell::strict`](crate::exec_endpoint::ExecCell::strict). Everything a
//! tenant can see comes from its grant or its sandbox.
//!
//! **Everything a tenant's prompt is made of comes from its sandbox.** [`ServiceSession::resources`]
//! reads skills, prompt templates, agent definitions, `AGENTS.md`/`CLAUDE.md` and
//! `SYSTEM.md`/`APPEND_SYSTEM.md` through the session's own exec backend, so every path advertised to
//! the model is one the model's own `read` can open, and no walk ever touches the replica. Session
//! start and `reload` call that one function, which is why `reload` is allowed here at all.
//!
//! **Its connectors come from its grant.** [`ServiceSession::mcp`] dials the grant's MCP list with the
//! sealed per-connector headers, per session, so one tenant's servers — and one tenant's elicitations
//! — never reach another's.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use crate::grant::{Grant, GrantVerifier};
use crate::session_store::{Layout, RepoOptions, TenantCodec};

/// Cap on the `x-beyond-grant` header's value. The whole header block is capped at 16 KiB
/// (`serve_ws::MAX_HEADER_BYTES`), so a token larger than this could never arrive beside a `Host`
/// and a `Sec-WebSocket-Key` anyway — rejecting it here means the crypto never runs on it.
pub const MAX_GRANT_BYTES: usize = 12 * 1024;

/// The header the edge relays a session grant in.
pub const GRANT_HEADER: &str = "x-beyond-grant";

/// The probe every service session runs before it touches a tool: the sandbox's own `$HOME`. It is
/// the smallest command that proves three things at once — the exec endpoint answers, its
/// credentials work, and there is a shell on the other side — and its output is exactly what
/// [`ShellFs::with_home`](crate::tools::fs::shell::ShellFs::with_home) needs to expand a `~`.
const HOME_PROBE: &str = r#"printf %s "$HOME""#;

/// How long the startup probe gets. Generous for one round trip to a sandbox that has to be awake
/// anyway, short enough that a dead endpoint fails the session rather than hanging the client.
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

// ---------------------------------------------------------------------------------------------
// Shards
// ---------------------------------------------------------------------------------------------

/// The tenant-data mounts this replica serves: `--shard <name>=</abs/path>`, repeatable.
///
/// A shard is a storage domain, not a tenant: many tenants live on one shard, each under its own
/// `<root>/<tenant>/` subtree. A session id minted for a shard carries it as a `<shard>.` prefix, so
/// an id alone says which mount to open — which is what lets one replica serve sessions from several
/// mounts without a lookup, and what makes a session for a mount this replica doesn't have a
/// **421** (ask another replica) rather than a 404.
#[derive(Debug, Default)]
pub struct Shards(BTreeMap<String, PathBuf>);

impl Shards {
    /// Parse `--shard` values. Each is `<name>=</absolute/path>`; the name is a path segment (so it
    /// can be a directory component and an id prefix) with no `.` (so `<shard>.<opaque>` splits
    /// unambiguously), and the path must be absolute — a relative one would resolve against the
    /// replica's cwd, which service mode has no business reading.
    pub fn parse(args: &[String]) -> Result<Self, String> {
        let mut map = BTreeMap::new();
        for arg in args {
            let (name, path) = arg
                .split_once('=')
                .ok_or_else(|| format!("--shard {arg:?}: expected <name>=</absolute/path>"))?;
            let (name, path) = (name.trim(), Path::new(path.trim()));
            if !is_shard_name(name) {
                return Err(format!(
                    "--shard {arg:?}: the name must be letters, digits, '-' or '_' (no '.', which \
                     separates the shard from the session id)"
                ));
            }
            if !path.is_absolute() {
                return Err(format!("--shard {arg:?}: the path must be absolute"));
            }
            if map.insert(name.to_owned(), path.to_path_buf()).is_some() {
                return Err(format!("--shard: {name:?} is given more than once"));
            }
        }
        Ok(Self(map))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&Path> {
        self.0.get(name).map(PathBuf::as_path)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Path)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v.as_path()))
    }

    /// Every mounted shard's directory for `tenant` — the set a tenant-scoped listing walks.
    pub fn tenant_dirs(&self, tenant: &str, leaf: &str) -> Vec<PathBuf> {
        self.0
            .values()
            .map(|root| root.join(tenant).join(leaf))
            .collect()
    }
}

/// The shard a session id names: the part before its first `.`, or `home` for an id with no prefix
/// (which is every id minted before the repo learns to prefix derived ids).
pub fn shard_of<'a>(session_id: &'a str, home: &'a str) -> &'a str {
    session_id.split_once('.').map_or(home, |(shard, _)| shard)
}

/// A shard name: a directory component *and* an id prefix, so no `.` and nothing path-special.
fn is_shard_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

/// A tenant id or connector name: one safe path segment. Same shape a session id must have
/// ([`is_valid_session_id`](crate::session_store::is_valid_session_id)), for the same reason — it
/// becomes part of a filesystem path.
fn is_safe_segment(s: &str) -> bool {
    s.len() <= 128 && crate::session_store::is_valid_session_id(s)
}

// ---------------------------------------------------------------------------------------------
// Authorization
// ---------------------------------------------------------------------------------------------

/// Why a connection was refused. One variant per HTTP status, so the connection path maps rather
/// than decides — an operator reading a 421 knows it is a routing problem and not an auth one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// 401 — no grant, or one that does not verify (unknown kid, bad signature, expired, unsealable).
    Unauthorized(&'static str),
    /// 400 — the request or the grant's own claims are unusable (no `?session_id=`, a tenant that
    /// isn't a path segment, a relative `workspace_root`).
    BadRequest(&'static str),
    /// 403 — the grant verifies, but is not for this session.
    Forbidden(&'static str),
    /// 421 Misdirected Request — the grant is fine; this replica just doesn't mount its shard. The
    /// edge should try another replica rather than the client retrying here.
    Misdirected(&'static str),
}

/// Unix seconds, failing **closed**: a clock the process cannot read at all yields `u64::MAX`, which
/// expires every grant, rather than `0`, which would accept every expired one.
pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(u64::MAX)
}

/// Verify `token` and bind it to `session_id`, producing this connection's [`ServiceSession`].
///
/// The order is the refusal table: a token that doesn't verify is a 401 before anything in it is
/// read; a grant that verifies but names another session is a 403; claims that aren't the right
/// *shape* are a 400; and only a well-formed grant for a shard this replica doesn't mount is a 421.
pub fn authorize(
    verifier: &GrantVerifier,
    shards: &Arc<Shards>,
    token: Option<&str>,
    session_id: Option<&str>,
    now: u64,
) -> Result<ServiceSession, Refusal> {
    let token = token.ok_or(Refusal::Unauthorized("a session grant is required"))?;
    if token.len() > MAX_GRANT_BYTES {
        return Err(Refusal::Unauthorized("the session grant is too large"));
    }
    let grant = verifier
        .verify(token, now)
        .map_err(|_| Refusal::Unauthorized("the session grant was rejected"))?;

    let session_id = session_id.ok_or(Refusal::BadRequest(
        "a session grant needs an explicit ?session_id=",
    ))?;
    if grant.session_id != session_id {
        return Err(Refusal::Forbidden(
            "the session grant is for a different session",
        ));
    }

    if !is_safe_segment(&grant.tenant) {
        return Err(Refusal::BadRequest("the session grant's tenant is invalid"));
    }
    if !is_shard_name(&grant.home_shard) {
        return Err(Refusal::BadRequest(
            "the session grant's home shard is invalid",
        ));
    }
    if !Path::new(&grant.workspace_root).is_absolute() {
        return Err(Refusal::BadRequest(
            "the session grant's workspace root must be absolute",
        ));
    }
    if !grant.mcp.iter().all(|c| is_safe_segment(&c.name)) {
        return Err(Refusal::BadRequest(
            "a session grant connector name is invalid",
        ));
    }

    let shard = shard_of(session_id, &grant.home_shard);
    if !is_shard_name(shard) {
        return Err(Refusal::BadRequest("the session id's shard is invalid"));
    }
    let session_root = shards.get(shard).ok_or(Refusal::Misdirected(
        "this replica does not mount that shard",
    ))?;
    let home_root = shards.get(&grant.home_shard).ok_or(Refusal::Misdirected(
        "this replica does not mount the grant's home shard",
    ))?;

    let session_dir = session_root.join(&grant.tenant).join("sessions");
    let memory_dir = home_root.join(&grant.tenant).join("memory");
    // Derived once per connection, from a key that never leaves this struct: `Debug` on the codec
    // shows only the tenant and the key fingerprint, and nothing ever writes the DEK itself down.
    let codec = Arc::new(TenantCodec::new(&grant.tenant, grant.secrets.dek.expose()));
    Ok(ServiceSession {
        grant: Arc::new(grant),
        shards: Arc::clone(shards),
        session_dir,
        memory_dir,
        codec,
        sandbox: OnceLock::new(),
    })
}

// ---------------------------------------------------------------------------------------------
// The session
// ---------------------------------------------------------------------------------------------

/// What the sandbox answered at session start. Recorded once, read by every tool build after it.
#[derive(Debug, Clone)]
pub struct Sandbox {
    /// The sandbox's `$HOME`, so a model-supplied `~/notes.md` expands against the tenant's own home
    /// rather than being left alone (or, far worse, resolved against the replica's).
    pub home: String,
    /// `bash` if the sandbox has one, else `sh`. Recorded here rather than probed per tool call.
    pub shell: &'static str,
}

/// One connection's verified grant, and everything derived from it that `serve` would otherwise have
/// taken from the replica host.
///
/// Shared by `Arc`: the connection that *spawns* a session hands its copy to that session's task,
/// where it fixes the session's secrets for the task's life (see the module doc). A later attach
/// still needs its own valid grant for the same `(tenant, session_id)` — that is what the
/// supervisor's per-slot tenant check enforces — but it does not re-point a live session's storage
/// or credentials.
pub struct ServiceSession {
    pub grant: Arc<Grant>,
    pub shards: Arc<Shards>,
    /// `<shard of the session id>/<tenant>/sessions` — where this session's transcript lives.
    pub session_dir: PathBuf,
    /// `<home shard>/<tenant>/memory` — the tenant's durable memory, shared by all of its sessions.
    pub memory_dir: PathBuf,
    /// This tenant's sealing keys, derived from the grant's per-tenant `dek`. Every transcript,
    /// listing cache and memory document this connection writes goes through it.
    codec: Arc<TenantCodec>,
    /// Filled by [`Self::connect_exec`] at session start; read by the tool builds after it.
    sandbox: OnceLock<Sandbox>,
}

impl ServiceSession {
    pub fn tenant(&self) -> &str {
        &self.grant.tenant
    }

    pub fn session_id(&self) -> &str {
        &self.grant.session_id
    }

    /// May this session use Code Mode? From the grant — outside service mode this is `--code-mode`,
    /// a per-process flag that service mode refuses because one tenant's setting must not be
    /// everybody's.
    pub fn code_mode(&self) -> bool {
        self.grant.code_mode
    }

    /// The tenant's workspace inside the sandbox — this session's `cwd` everywhere `serve` reports
    /// or records one.
    pub fn workspace_root(&self) -> &str {
        &self.grant.workspace_root
    }

    /// The grant's gateway credential. Service mode never resolves a credential any other way: no
    /// `--key`, no `models.json`, no OAuth store, no ambient provider env.
    pub fn gateway_key(&self) -> &str {
        self.grant.secrets.gateway_key.expose()
    }

    /// The sandbox, once probed. `None` before [`Self::connect_exec`] has run — no tool can run in
    /// that window, because the session fails to start if the probe fails.
    pub fn sandbox(&self) -> Option<&Sandbox> {
        self.sandbox.get()
    }

    /// This tenant's session directories across **every** mounted shard — what a tenant-scoped
    /// listing walks, so a session minted on another shard is still listable here.
    pub fn tenant_session_dirs(&self) -> Vec<PathBuf> {
        self.shards.tenant_dirs(self.tenant(), "sessions")
    }

    /// The directory holding `id`, by `id`'s own shard prefix. `None` when that shard isn't mounted
    /// — the caller reports it rather than silently operating on the wrong mount.
    pub fn session_dir_for(&self, id: &str) -> Option<PathBuf> {
        let shard = shard_of(id, &self.grant.home_shard);
        Some(self.shards.get(shard)?.join(self.tenant()).join("sessions"))
    }

    /// The shard this session's transcript lives on — its id's own `<shard>.` prefix, or the grant's
    /// home shard for an id minted before ids carried one.
    pub fn shard(&self) -> &str {
        shard_of(self.session_id(), &self.grant.home_shard)
    }

    /// This tenant's sealing keys. Shared rather than re-derived: a listing, a fork and a preview all
    /// open *other* ids, and they all seal under the same per-tenant key.
    pub fn codec(&self) -> &Arc<TenantCodec> {
        &self.codec
    }

    /// The on-disk shape of every session this tenant owns: epoch segments (so two replicas sharing a
    /// mount fence each other rather than interleaving), sealed with this tenant's keys.
    pub fn layout(&self) -> Layout {
        Layout::Segmented {
            codec: Some(Arc::clone(&self.codec)),
        }
    }

    /// How to open the repo holding `id`. The id prefix is `id`'s **own** shard, so a session derived
    /// from it (a fork, a clone, an archive) is minted onto the mount its parent already lives on and
    /// stays routable by its id alone.
    pub fn repo_options_for(&self, id: &str) -> RepoOptions {
        RepoOptions {
            layout: self.layout(),
            id_prefix: Some(shard_of(id, &self.grant.home_shard).to_owned()),
        }
    }

    /// This session's own directory — the unit [`acquire_session_lock`](crate::session_store::acquire_session_lock)
    /// takes a lock on, and the parent of its segments and its `/session` memory.
    pub fn session_path(&self) -> PathBuf {
        self.session_dir.join(self.session_id())
    }

    /// Durable memory, rooted at the tenant's own directory rather than resolved from the replica's
    /// cwd or `--memory` DSN, and sealed with the tenant's own key — the mount is shared with every
    /// other tenant on the shard.
    pub fn memory_backend(&self) -> Arc<dyn crate::memory::MemoryBackend> {
        Arc::new(
            crate::memory::file::FileBackend::at(self.memory_dir.clone())
                .sealed(Arc::clone(&self.codec)),
        )
    }

    /// Build this session's exec target from the grant, **strictly**: the endpoint is probed before
    /// any tool can reach it, and a failure is returned to the caller, which ends the session.
    ///
    /// The probe is `sh -c 'printf %s "$HOME"'`. Its output becomes the backend's home (so `~`
    /// expands inside the sandbox, not on the replica), and a second `command -v bash` records which
    /// shell the remote `bash` tool should ask for. Both are cached on this `ServiceSession`.
    pub async fn connect_exec(
        &self,
        max_response_bytes: usize,
    ) -> Result<crate::exec_endpoint::ExecTarget, String> {
        use crate::exec_endpoint::{ExecTarget, HttpExecRunner};
        use crate::tools::exec::CommandRunner;

        let mut runner = HttpExecRunner::new(self.grant.exec_url.as_str())?
            .with_max_response_bytes(max_response_bytes);
        for header in &self.grant.secrets.exec_headers {
            runner = runner.with_header(header.name.as_str(), header.value.expose());
        }
        let runner: Arc<dyn CommandRunner> = Arc::new(runner);

        let home = runner
            .run(
                "sh",
                &["-c".to_string(), HOME_PROBE.to_string()],
                None,
                PROBE_TIMEOUT,
            )
            .await
            .map_err(|e| format!("the sandbox did not answer: {e}"))?;
        if home.code != Some(0) || home.stdout.trim().is_empty() {
            return Err(format!(
                "the sandbox did not report a home directory (exit {:?})",
                home.code
            ));
        }
        let home = home.stdout.trim().to_string();
        // The remote `bash` tool prefers a real bash and falls back to `sh`, which every sandbox has
        // by construction (the probe above just ran one).
        let shell = runner
            .run(
                "command",
                &["-v".to_string(), "bash".to_string()],
                None,
                PROBE_TIMEOUT,
            )
            .await
            .is_ok_and(|r| r.code == Some(0) && !r.stdout.trim().is_empty());
        let _ = self.sandbox.set(Sandbox {
            home: home.clone(),
            shell: if shell { "bash" } else { "sh" },
        });
        Ok(ExecTarget::over_with_home(runner, Some(home)).await)
    }

    /// What a tenant's own sandbox contributes to the prompt: its skills, prompt templates, agent
    /// definitions, `AGENTS.md`/`CLAUDE.md` and `SYSTEM.md`/`APPEND_SYSTEM.md` — every one of them
    /// read **through `backend`**, which is the session's exec endpoint, so every path named is a
    /// sandbox path and no walk touches the replica.
    ///
    /// This is the seam both session start and `reload` call, so the two cannot drift: `reload` is
    /// re-enabled in service mode precisely because "re-walk the filesystem" now means the tenant's
    /// own, and re-running this is the whole of it.
    ///
    /// The roots are the sandbox spellings of the ones [`skills::discover`](crate::skills::discover)
    /// uses, in ascending specificity — the tenant's `$HOME` first, then its workspace, so a
    /// workspace-local definition shadows a home-wide one of the same name. They are **not**
    /// trust-gated: on this host that gate protects an operator from a checkout they did not write,
    /// and inside one tenant's own box there is no second party to protect from.
    ///
    /// `context_files` is the caller's `--no-context-files` answer, honored here rather than after
    /// the fact so a session that does not want them pays no round trips for them.
    pub async fn resources(
        &self,
        backend: &dyn crate::tools::fs::FsBackend,
        context_files: bool,
    ) -> Resources {
        let root = Path::new(self.workspace_root());
        let home = self.sandbox().map(|s| Path::new(s.home.as_str()));
        let (skills, skill_collisions) =
            crate::skills::discover_via(backend, &skill_roots(root, home)).await;
        let (prompt_templates, prompt_collisions) =
            crate::prompts::discover_via(backend, &claude_roots(root, home, "prompts")).await;
        let (agents, _agent_collisions) =
            crate::agents::discover_via(backend, &claude_roots(root, home, "agents")).await;
        Resources {
            skills,
            skill_collisions,
            prompt_templates,
            prompt_collisions,
            agents,
            context_files: match context_files {
                true => crate::resources::load_context_files_via(backend, root, home).await,
                false => Vec::new(),
            },
            system: crate::resources::claude_file_via(backend, root, home, "SYSTEM.md").await,
            append_system: crate::resources::claude_file_via(
                backend,
                root,
                home,
                "APPEND_SYSTEM.md",
            )
            .await,
        }
    }

    /// Dial this session's own MCP connectors — the ones its grant names, with the credentials its
    /// grant sealed — and return their tools and catalog.
    ///
    /// Per session, not per process: the replica's configured servers are the operator's and are
    /// never connected in service mode at all, and two sessions on this replica share no MCP state.
    /// `host` is this session's own [`McpHost`](crate::tools::mcp_host::McpHost), so an elicitation
    /// or a sampling request from one of these servers is answered by the session that asked.
    ///
    /// `egress` is the daemon's process-wide, SSRF-checked client (`ServeConfig::mcp_http`). Without
    /// one there is nothing safe to dial through, so the session simply has no connectors — the same
    /// fail-closed answer as a connector whose URL is refused.
    pub async fn mcp(
        &self,
        egress: Option<&crate::tools::mcp::McpEgress>,
        host: Arc<crate::tools::mcp_host::McpHost>,
    ) -> (
        Vec<Arc<dyn agent_core::Tool>>,
        crate::tools::mcp::McpCatalog,
        Vec<String>,
    ) {
        let empty = || (Vec::new(), crate::tools::mcp::McpCatalog::default());
        if self.grant.mcp.is_empty() {
            let (tools, catalog) = empty();
            return (tools, catalog, Vec::new());
        }
        let Some(egress) = egress else {
            // Never in a real service replica (`main.rs` builds one whenever `--service` is set),
            // so this says so rather than dropping the connectors silently.
            let (tools, catalog) = empty();
            return (
                tools,
                catalog,
                vec![
                    "this replica has no MCP egress client, so the grant's connectors were not \
                     dialed"
                        .to_owned(),
                ],
            );
        };
        crate::tools::mcp::connect_granted(
            &self.grant.mcp,
            &self.grant.secrets.mcp_headers,
            egress,
            host,
            crate::tools::mcp::idle_reap_after_from_env(),
        )
        .await
    }

    /// Strip replica-host paths out of `text` — an error from the storage layer names the file it
    /// failed on, and a tenant has no business learning this replica's mount layout. Shard roots
    /// become `<store>`; the replica's own `$HOME`/cwd become `<host>`.
    pub fn redact(&self, text: &str) -> String {
        let mut out = text.to_owned();
        for (_, root) in self.shards.iter() {
            let root = root.to_string_lossy();
            if !root.is_empty() && out.contains(root.as_ref()) {
                out = out.replace(root.as_ref(), "<store>");
            }
        }
        for host in [
            std::env::var_os("HOME").map(PathBuf::from),
            std::env::current_dir().ok(),
        ]
        .into_iter()
        .flatten()
        {
            let host = host.to_string_lossy().into_owned();
            if host.len() > 1 && out.contains(&host) {
                out = out.replace(&host, "<host>");
            }
        }
        out
    }
}

/// The sandbox discovery roots for skills, in ascending specificity (a later root's same-named skill
/// wins). `.agents/skills` before `.claude/skills` for the same reason the on-host walk orders them
/// that way: the tool-specific directory is the one written deliberately for this agent.
fn skill_roots(root: &Path, home: Option<&Path>) -> Vec<(PathBuf, &'static str)> {
    let mut roots = Vec::with_capacity(3);
    if let Some(home) = home {
        roots.push((home.join(".claude/skills"), "user"));
    }
    roots.push((root.join(".agents/skills"), "project"));
    roots.push((root.join(".claude/skills"), "project"));
    roots
}

/// The sandbox discovery roots for a flat `.claude/<leaf>` resource kind (`agents`, `prompts`), in
/// the same ascending order.
fn claude_roots(root: &Path, home: Option<&Path>, leaf: &str) -> Vec<(PathBuf, &'static str)> {
    let mut roots = Vec::with_capacity(2);
    if let Some(home) = home {
        roots.push((home.join(".claude").join(leaf), "user"));
    }
    roots.push((root.join(".claude").join(leaf), "project"));
    roots
}

/// Everything sandbox discovery contributes to a session, as one value so the wiring is written
/// once — and so `reload` refreshes exactly the same set session start built.
#[derive(Default)]
pub struct Resources {
    pub skills: Vec<crate::skills::Skill>,
    /// Shadowed skill names and unreadable manifests, surfaced through `get_commands` exactly as the
    /// on-host walk's are — a tenant debugging its own sandbox gets the same signal.
    pub skill_collisions: Vec<crate::skills::Collision>,
    pub prompt_templates: Vec<crate::prompts::PromptTemplate>,
    pub prompt_collisions: Vec<crate::skills::Collision>,
    pub agents: Vec<crate::agents::AgentDef>,
    /// The tenant's own `AGENTS.md`/`CLAUDE.md`, already read: `(path, body)` in prompt order.
    pub context_files: Vec<(String, String)>,
    /// `<workspace>/.claude/SYSTEM.md`, else `<sandbox home>/.claude/SYSTEM.md`. Delivered as
    /// [`PromptOptions::base`](crate::resources::PromptOptions::base), never through
    /// `disk_overrides` — see that field's doc comment.
    pub system: Option<String>,
    /// `APPEND_SYSTEM.md`, same lookup. An explicit `--append-system-prompt` outranks it.
    pub append_system: Option<String>,
}

// ---------------------------------------------------------------------------------------------
// Command gating
// ---------------------------------------------------------------------------------------------

/// Why a control command is refused in service mode, or `None` if it is allowed.
///
/// Two groups. The first are host operations that have no tenant meaning at all and would, if left
/// on, reach the replica: re-pointing the exec endpoint (the grant decides that), the whole
/// interactive login surface (an operator's own credential store), and switching to a session this
/// connection has no grant for. `reload` is **not** among them any more — it re-runs
/// [`ServiceSession::resources`], which walks the tenant's sandbox, so it means here exactly what it
/// means anywhere else.
///
/// `fork`, `clone` and `new_session` are **not** here: a repo opened for a tenant mints derived ids
/// as `<shard>.<opaque>` ([`ServiceSession::repo_options_for`]), so a derived session stays routable
/// on the mount its parent lives on.
pub fn refused_command(command: &str) -> Option<&'static str> {
    Some(match command {
        "set_exec_endpoint" => {
            "refused in service mode: the session grant fixes this session's exec endpoint"
        }
        "login" | "submit_code" | "abort_login" | "logout" | "auth_status" => {
            "refused in service mode: credentials come from the session grant"
        }
        "switch_session" => {
            "refused in service mode: connect at ?session_id=<id> with a grant for that session"
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shards(pairs: &[(&str, &str)]) -> Arc<Shards> {
        Arc::new(
            Shards::parse(
                &pairs
                    .iter()
                    .map(|(n, p)| format!("{n}={p}"))
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn shard_flags_demand_a_name_and_an_absolute_path() {
        assert!(Shards::parse(&["a=/mnt/a".into()]).is_ok());
        assert!(Shards::parse(&["a".into()]).is_err(), "no '='");
        assert!(Shards::parse(&["a=rel".into()]).is_err(), "relative");
        assert!(
            Shards::parse(&["a.b=/mnt/a".into()]).is_err(),
            "dot in name"
        );
        assert!(Shards::parse(&["=/mnt/a".into()]).is_err(), "empty name");
        assert!(
            Shards::parse(&["a=/mnt/a".into(), "a=/mnt/b".into()]).is_err(),
            "duplicate"
        );
    }

    #[test]
    fn an_id_names_its_own_shard_and_an_unprefixed_one_falls_to_home() {
        assert_eq!(shard_of("s1.abc", "home"), "s1");
        assert_eq!(shard_of("abc", "home"), "home");
        assert_eq!(shard_of("s1.a.b", "home"), "s1");
    }

    #[test]
    fn tenant_dirs_cover_every_mounted_shard() {
        let s = shards(&[("a", "/mnt/a"), ("b", "/mnt/b")]);
        let dirs = s.tenant_dirs("t1", "sessions");
        assert_eq!(
            dirs,
            vec![
                PathBuf::from("/mnt/a/t1/sessions"),
                PathBuf::from("/mnt/b/t1/sessions"),
            ]
        );
    }

    #[test]
    fn every_host_command_that_could_reach_the_replica_is_refused() {
        for command in [
            "set_exec_endpoint",
            "login",
            "submit_code",
            "abort_login",
            "logout",
            "auth_status",
            "switch_session",
        ] {
            assert!(refused_command(command).is_some(), "{command}");
        }
        for command in [
            "prompt",
            "get_state",
            "list_sessions",
            "export_html",
            "bash",
            // Derived sessions are minted onto their parent's shard, so these are allowed.
            "fork",
            "clone",
            "new_session",
            // Re-enabled by sandbox discovery: it re-walks the *tenant's* filesystem now.
            "reload",
        ] {
            assert!(refused_command(command).is_none(), "{command}");
        }
    }

    /// A `ServiceSession` with no connection behind it, for the derivations that are pure functions
    /// of the grant.
    fn session(mounts: &[(&str, &str)], session_id: &str) -> ServiceSession {
        let grant = Grant {
            tenant: "t1".into(),
            session_id: session_id.into(),
            home_shard: "a".into(),
            workspace_root: "/w".into(),
            exec_url: "http://x/".into(),
            mcp: Vec::new(),
            exp: 0,
            code_mode: false,
            secrets: serde_json::from_str(
                r#"{"exec_headers":[],"mcp_headers":{},"gateway_key":"k","dek":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="}"#,
            )
            .unwrap(),
        };
        let codec = Arc::new(TenantCodec::new(&grant.tenant, grant.secrets.dek.expose()));
        ServiceSession {
            grant: Arc::new(grant),
            shards: shards(mounts),
            session_dir: PathBuf::from("/mnt/efs/a/t1/sessions"),
            memory_dir: PathBuf::from("/mnt/efs/a/t1/memory"),
            codec,
            sandbox: OnceLock::new(),
        }
    }

    #[test]
    fn redaction_removes_the_mount_layout_from_an_error() {
        let session = session(&[("a", "/mnt/efs/a")], "a.x");
        let redacted =
            session.redact("failed to open /mnt/efs/a/t1/sessions/1_x.jsonl: No such file");
        assert!(!redacted.contains("/mnt/efs"), "{redacted}");
        assert!(redacted.contains("<store>/t1/sessions"), "{redacted}");
    }

    #[test]
    fn a_repo_mints_derived_ids_onto_the_shard_the_parent_lives_on() {
        let session = session(&[("a", "/mnt/efs/a"), ("b", "/mnt/efs/b")], "a.x");
        assert_eq!(session.shard(), "a");
        assert_eq!(
            session.repo_options_for("a.x").id_prefix.as_deref(),
            Some("a")
        );
        // A command naming a session on another mount derives onto *that* mount.
        assert_eq!(
            session.repo_options_for("b.y").id_prefix.as_deref(),
            Some("b")
        );
        // An id with no prefix at all belongs to the grant's home shard.
        assert_eq!(
            session.repo_options_for("bare").id_prefix.as_deref(),
            Some("a")
        );
        assert!(matches!(
            session.layout(),
            Layout::Segmented { codec: Some(_) }
        ));
    }

    #[test]
    fn the_session_lock_is_taken_on_the_sessions_own_directory() {
        let session = session(&[("a", "/mnt/efs/a")], "a.x");
        assert_eq!(
            session.session_path(),
            PathBuf::from("/mnt/efs/a/t1/sessions/a.x"),
        );
    }
}
